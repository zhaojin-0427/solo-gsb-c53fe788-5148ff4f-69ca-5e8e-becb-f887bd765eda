"""印后测色验收（post-press colorimetric acceptance）.

对一个**已保存软打样任务**（可再绑定一个**已达标修色成品**），把用户上传
的逐点实测 CSV（零基整数坐标 ``x,y`` 与有限 D50 ``L,a,b``）与沿**原任务
印刷 ICC、原任务意图**回算出的预测 Lab 逐点计算 **CIEDE2000**：

* **不绑定修色**：全部测点的预测 Lab 即原任务 ``lab_press.npy``（分色
  CMYK 经印刷 ICC 回算的逐像素预测值）；
* **绑定达标修色成品**：修色报告只有**选区平均 Lab**，不足以逐点验收，
  因此以已核验的成品为据——验收先独立核验成品 TIFF 的全部绑定（可打开、
  CMYK、尺寸、嵌入印刷 ICC 摘要、选区外与原分色逐字节相同、选区内恰为
  报告记录的统一偏移），核验通过后**按原分色 + 报告偏移重建**修色测点
  CMYK，再沿原印刷 ICC 与原任务 ``source→press`` 意图回算 Lab；
  **选区外测点保持原任务预测值** ``lab_press.npy``，绝不以原任务值代替
  修色选区内的预测。

同坐标复测**逐条保留**（不去重、不平均）；**任一测点 ΔE00 大于上限即
整张不合格**。产物按完整请求（任务、可选修色、规范测点序列、ΔE 上限）
内容寻址，报告不含时间戳，相同输入逐字节复用。

任何下列情况一律拒绝验收（HTTP 422/404），绝不在错误的绑定上出结论：
修色不可达、修色报告或成品与任务/选区/偏移/ICC 绑定不符、原任务印刷
ICC 缺失或摘要不符、测点坐标越界、CSV 非法（含空文件、非有限 Lab 等）。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import colorscience, lcms_native

# 与 ΔE 判定保持一致的浮点容差：de == limit 视为不超限（“大于上限才不合格”）。
DE_EPS = 1e-9


class AcceptanceError(ValueError):
    """请求参数 / 绑定非法（HTTP 422）。"""


class AcceptanceNotFound(AcceptanceError):
    """引用的任务 / 修色 / 产物不存在（HTTP 404）。"""


@dataclass(frozen=True)
class Measurement:
    """一条 CSV 测点（零基整数坐标 + 有限 D50 Lab）。"""
    x: int
    y: int
    L: float
    a: float
    b: float


@dataclass(frozen=True)
class AcceptanceRequest:
    job_id: str
    retouch_id: str | None
    limit_de: float
    points: tuple[Measurement, ...]
    csv_canonical: str


# ---------------------------------------------------------------------------
# CSV 解析与严格校验
# ---------------------------------------------------------------------------

_CSV_HEADER = ("x", "y", "l", "a", "b")


def parse_measurements_csv(raw: bytes) -> AcceptanceRequest:
    """解析并严格校验测点 CSV；任何不合法输入抛 AcceptanceError。

    规则：UTF-8 文本；首行表头必须且只能是 ``x,y,L,a,b``（列名大小写不
    敏感、允许首尾空白与 UTF-8 BOM）；至少一条数据行；每行恰好 5 列；
    ``x/y`` 为零基非负整数（拒绝 ``1.0`` 之类浮点与布尔）；``L/a/b`` 为
    有限数值，``L ∈ [0,100]``、``a,b ∈ [-128,127]``。空行、引号/转义
    错误、列数不符、非有限/越界值一律拒绝。同坐标复测允许重复，逐条保留。
    """
    if not isinstance(raw, (bytes, bytearray)) or len(raw) == 0:
        raise AcceptanceError("测点 CSV 为空：必须包含表头与至少一条测点")
    try:
        text = bytes(raw).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AcceptanceError(f"测点 CSV 不是合法的 UTF-8 文本: {exc}")

    if any(ch in text for ch in ("\r",)):
        # 统一接受 CRLF / LF：csv 模块本身能处理 \r\n，但单独的 \r 也规整化。
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    try:
        reader = csv.reader(io.StringIO(text, newline=None))
        rows = list(reader)
    except csv.Error as exc:
        raise AcceptanceError(f"测点 CSV 解析失败: {exc}")

    if not rows:
        raise AcceptanceError("测点 CSV 为空：必须包含表头与至少一条测点")
    header = tuple(c.strip().lower() for c in rows[0])
    if header != _CSV_HEADER:
        raise AcceptanceError(
            "测点 CSV 表头必须恰好为 x,y,L,a,b（首行、5 列、列名顺序固定），"
            f"实际为 {','.join(rows[0]) if rows[0] else '（空行）'}"
        )

    points: list[Measurement] = []
    for index, row in enumerate(rows[1:], start=2):
        if len(row) == 0 or (len(row) == 1 and row[0].strip() == ""):
            raise AcceptanceError(f"测点 CSV 第 {index} 行为空行，拒绝验收")
        if len(row) != 5:
            raise AcceptanceError(
                f"测点 CSV 第 {index} 行有 {len(row)} 列，必须恰好为 5 列 x,y,L,a,b"
            )
        cells = [c.strip() for c in row]
        x = _parse_int_cell(cells[0], "x", index)
        y = _parse_int_cell(cells[1], "y", index)
        L = _parse_lab_cell(cells[2], "L", index, 0.0, 100.0)
        a = _parse_lab_cell(cells[3], "a", index, -128.0, 127.0)
        b = _parse_lab_cell(cells[4], "b", index, -128.0, 127.0)
        points.append(Measurement(x=x, y=y, L=L, a=a, b=b))

    if not points:
        raise AcceptanceError("测点 CSV 没有任何测点（至少需要一条数据行）")

    # 规范化 CSV：与解析结果严格对应、行序不变，作为报告绑定与确定性复用的
    # 凭据；坐标为整数、Lab 走最短往返浮点表示（-0 归一为 0）。
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["x", "y", "L", "a", "b"])
    for p in points:
        writer.writerow([p.x, p.y, _canon_float(p.L), _canon_float(p.a),
                         _canon_float(p.b)])
    return AcceptanceRequest(
        job_id="", retouch_id=None, limit_de=0.0,
        points=tuple(points), csv_canonical=out.getvalue(),
    )


def _parse_int_cell(cell: str, name: str, line: int) -> int:
    # 拒绝 "1.0"、"+1"、"0x1"、Unicode 全角/阿拉伯-印度数字、内嵌空格等：
    # 只接受 ASCII 十进制整数字面量（零基、非负）。
    if not cell or not all("0" <= ch <= "9" for ch in cell):
        raise AcceptanceError(
            f"测点 CSV 第 {line} 行 {name}={cell!r} 不是零基非负整数"
        )
    return int(cell)


def _parse_lab_cell(cell: str, name: str, line: int,
                    lo: float, hi: float) -> float:
    if cell == "":
        raise AcceptanceError(f"测点 CSV 第 {line} 行 {name} 缺失")
    try:
        value = float(cell)
    except ValueError:
        raise AcceptanceError(f"测点 CSV 第 {line} 行 {name}={cell!r} 不是数值")
    if not np.isfinite(value):
        raise AcceptanceError(
            f"测点 CSV 第 {line} 行 {name}={cell} 不是有限数值（拒绝 NaN/Inf）"
        )
    if not (lo <= value <= hi):
        raise AcceptanceError(
            f"测点 CSV 第 {line} 行 {name}={value} 超出允许范围 [{lo}, {hi}]"
        )
    return float(value)


def validate_request(raw_points: AcceptanceRequest, job_id: str,
                     retouch_id: str | None, limit_de: float) -> AcceptanceRequest:
    """组合并校验一次验收请求的非 CSV 参数。"""
    if not isinstance(job_id, str) or not job_id:
        raise AcceptanceError("缺少已保存任务 job_id")
    if retouch_id is not None and (not isinstance(retouch_id, str)
                                   or not retouch_id):
        raise AcceptanceError("retouch_id 必须是非空字符串或省略")
    if isinstance(limit_de, bool) or not isinstance(limit_de, (int, float)):
        raise AcceptanceError("ΔE00 上限必须是非负数值")
    limit = float(limit_de)
    if not np.isfinite(limit) or limit < 0.0:
        raise AcceptanceError("ΔE00 上限必须是非负有限数值")
    return AcceptanceRequest(
        job_id=job_id,
        retouch_id=retouch_id,
        limit_de=limit,
        points=raw_points.points,
        csv_canonical=raw_points.csv_canonical,
    )


# ---------------------------------------------------------------------------
# 绑定核验
# ---------------------------------------------------------------------------

def _recorded_press_digest(conn, job_id: str, meta: dict) -> str | None:
    """取任务创建时固化的印刷 ICC 全 64 位 SHA-256（数据库优先，meta 兜底）。"""
    digest = None
    if conn is not None:
        try:
            from . import database as db
            row = db.get_job(conn, job_id)
            if row is not None and row["profile_digests_json"]:
                digest = json.loads(
                    row["profile_digests_json"]).get("press")
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
            digest = None
    return digest or meta.get("press_profile_sha256")


def verify_job_press_icc(meta: dict, press_bytes: bytes,
                         recorded_digest: str | None) -> None:
    """原任务印刷 ICC：摘要记录必须存在、当前字节可打开且摘要严格一致。"""
    if not recorded_digest:
        raise AcceptanceError("原任务缺少印刷 ICC 摘要记录，拒绝验收")
    digest = hashlib.sha256(press_bytes).hexdigest()
    if digest != recorded_digest:
        raise AcceptanceError(
            "原任务印刷 ICC 摘要不符：任务记录为 "
            f"{recorded_digest}，当前文件为 {digest}；拒绝在被替换的配置上验收"
        )
    try:
        lcms_native.read_header(press_bytes)
    except Exception as exc:
        raise AcceptanceError(f"原任务印刷 ICC 已损坏，无法打开: {exc}")


def verify_retouch_binding(retouch_id: str, retouch_dir: Path,
                           cmyk_full: np.ndarray, meta: dict,
                           press_bytes: bytes) -> dict:
    """核验一个“达标修色成品”可作为验收依据，返回其报告。

    拒绝条件（任一不符即 422，绝不在错误成品上回算）：
    * 修色目录/报告不存在（404）；报告不可达或缺少成品 TIFF；
    * 修色报告绑定的原任务 job_id 与本次任务不同；
    * 修色记录的印刷 ICC 摘要与当前原任务印刷 ICC 字节不符；
    * 成品不可打开、非 8 位 CMYK、尺寸与原任务不一致、嵌入 ICC 字节不符；
    * 成品选区外与原任务分色 ``cmyk.npy`` 不逐字节相同，或选区内不是报告
      记录的那一个统一整数偏移。
    """
    base = retouch_dir / retouch_id
    report_path = base / "report.json"
    if not report_path.exists():
        raise AcceptanceNotFound("修色任务不存在或其报告已被删除")
    try:
        report = json.loads(report_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AcceptanceError(f"修色报告不可读，拒绝作为验收依据: {exc}")
    if not isinstance(report, dict):
        raise AcceptanceError("修色报告结构非法，拒绝验收")

    if not report.get("reachable"):
        raise AcceptanceError("修色未达标（报告标记为不可达），不能作为印后验收依据")
    if report.get("source_job", {}).get("job_id") != meta.get("job_id"):
        raise AcceptanceError("修色报告绑定的原任务与所选任务不符，拒绝验收")

    press_digest = hashlib.sha256(press_bytes).hexdigest()
    press_summary = report.get("profiles", {}).get("press", {})
    if press_summary.get("sha256") != press_digest:
        raise AcceptanceError(
            "修色报告绑定的印刷 ICC 摘要与原任务当前印刷 ICC 不符，拒绝验收"
        )

    region = report.get("region")
    offset = report.get("best_offset_cmyk")
    if not (isinstance(region, dict)
            and all(isinstance(region.get(k), int) for k in ("x", "y", "width", "height"))):
        raise AcceptanceError("修色报告的选区矩形非法，拒绝验收")
    if not (isinstance(offset, list) and len(offset) == 4
            and all(isinstance(v, int) and not isinstance(v, bool) for v in offset)):
        raise AcceptanceError("修色报告的统一偏移非法，拒绝验收")

    tiff_path = base / "retouched.tif"
    if not tiff_path.is_file():
        raise AcceptanceError("达标修色成品 retouched.tif 缺失，拒绝验收")
    try:
        with Image.open(tiff_path) as im:
            im.load()
            if im.mode != "CMYK":
                raise AcceptanceError("修色成品不是 CMYK TIFF，拒绝验收")
            if im.size != (int(cmyk_full.shape[1]), int(cmyk_full.shape[0])):
                raise AcceptanceError("修色成品尺寸与原任务不一致，拒绝验收")
            embedded = im.info.get("icc_profile")
            tiff_cmyk = np.asarray(im)
    except AcceptanceError:
        raise
    except Exception as exc:
        raise AcceptanceError(f"修色成品无法打开，拒绝验收: {exc}")
    if embedded != press_bytes:
        raise AcceptanceError("修色成品嵌入的印刷 ICC 与原任务印刷 ICC 字节不符，拒绝验收")

    if tiff_cmyk.shape != cmyk_full.shape:
        raise AcceptanceError("修色成品像素形状与原任务分色不一致，拒绝验收")

    x, y = region["x"], region["y"]
    rw, rh = region["width"], region["height"]
    x2, y2 = x + rw, y + rh
    H, W = cmyk_full.shape[:2]
    if not (0 <= x < W and 0 <= y < H and x2 <= W and y2 <= H):
        raise AcceptanceError("修色报告的选区超出原任务图像范围，拒绝验收")

    roi_mask = np.zeros(cmyk_full.shape[:2], dtype=bool)
    roi_mask[y:y2, x:x2] = True

    # 选区外：成品必须与原任务分色逐字节相同。
    if not np.array_equal(tiff_cmyk[~roi_mask], cmyk_full[~roi_mask]):
        raise AcceptanceError(
            "修色成品选区外字节与原任务分色不一致（成品绑定不符），拒绝验收"
        )
    # 选区内：必须恰好等于原分色 + 报告记录的统一偏移（逐像素、逐通道）。
    off = np.asarray(offset, dtype=np.int32)
    expected_roi = np.clip(
        cmyk_full[y:y2, x:x2].astype(np.int32) + off[None, None, :], 0, 255
    ).astype(np.uint8)
    if not np.array_equal(tiff_cmyk[y:y2, x:x2], expected_roi):
        raise AcceptanceError(
            "修色成品选区内像素与报告记录的统一偏移不符（成品绑定不符），拒绝验收"
        )

    return report


# ---------------------------------------------------------------------------
# 预测 Lab 回算与 ΔE00 验收
# ---------------------------------------------------------------------------

def evaluate(req: AcceptanceRequest, meta: dict, cmyk_full: np.ndarray,
             lab_press: np.ndarray, press_bytes: bytes,
             retouch_report: dict | None) -> dict:
    """逐点回算预测 Lab、计算 ΔE00 并判定整张合格性。

    选区内（绑定修色时）的预测由“原分色 + 统一偏移”重建 CMYK 后，沿原
    印刷 ICC 与原任务 source→press 意图批量回算；选区外（以及不绑定修色
    时的全部点）直接使用原任务逐像素预测 ``lab_press``。同坐标复测逐条
    保留，结果行序与 CSV 行序一致。
    """
    height, width = lab_press.shape[:2]
    xs = np.array([p.x for p in req.points], dtype=np.int64)
    ys = np.array([p.y for p in req.points], dtype=np.int64)
    if np.any(xs >= width) or np.any(ys >= height):
        bad = [
            f"({p.x},{p.y})" for p in req.points
            if p.x >= width or p.y >= height
        ]
        raise AcceptanceError(
            f"测点坐标越界（任务图像尺寸 {width}×{height}，坐标零基、"
            f"x/y 须分别 < {width}/{height}），越界测点：{', '.join(bad[:10])}"
            + (" …" if len(bad) > 10 else "")
        )

    in_roi = np.zeros(len(req.points), dtype=bool)
    if retouch_report is not None:
        r = retouch_report["region"]
        in_roi = (
            (xs >= r["x"]) & (xs < r["x"] + r["width"])
            & (ys >= r["y"]) & (ys < r["y"] + r["height"])
        )

    predicted = np.asarray(lab_press[ys, xs], dtype=np.float64).copy()

    if in_roi.any():
        intent = meta.get("source_intent", "relative_colorimetric")
        if intent not in lcms_native.INTENTS:
            raise AcceptanceError(f"原任务渲染意图无效: {intent}")
        off = np.asarray(retouch_report["best_offset_cmyk"], dtype=np.int32)
        idx = np.nonzero(in_roi)[0]
        base_cmyk = np.asarray(cmyk_full[ys[idx], xs[idx]], dtype=np.int32)
        shifted = np.clip(base_cmyk + off[None, :], 0, 255).astype(np.uint8)
        try:
            with lcms_native.lab_transform("CMYK", press_bytes, intent) as xf:
                predicted[idx] = xf(np.ascontiguousarray(shifted))
        except RuntimeError as exc:
            raise AcceptanceError(f"沿原印刷 ICC 回算修色测点 Lab 失败: {exc}")

    measured = np.array([[p.L, p.a, p.b] for p in req.points], dtype=np.float64)
    de = colorscience.ciede2000(
        measured.reshape(-1, 1, 3), predicted.reshape(-1, 1, 3)
    ).reshape(-1)

    limit = float(req.limit_de)
    over = de > limit + DE_EPS
    points_out = []
    for i, p in enumerate(req.points):
        points_out.append({
            "index": i + 1,
            "x": int(p.x),
            "y": int(p.y),
            "measured_lab": [round(float(p.L), 4), round(float(p.a), 4),
                             round(float(p.b), 4)],
            "predicted_lab": [round(float(v), 4) for v in predicted[i]],
            "de00": round(float(de[i]), 4),
            "within_limit": not bool(over[i]),
            "source": ("retouched_region" if in_roi[i]
                       else "job_lab_press"),
        })

    over_coords = [[int(p.x), int(p.y)] for p, o in zip(req.points, over) if o]
    return {
        "passed": bool(not over.any()),
        "limit_de00": limit,
        "point_count": int(len(req.points)),
        "over_count": int(over.sum()),
        "max_de00": round(float(de.max()), 4),
        "mean_de00": round(float(de.mean()), 4),
        "over_coordinates": over_coords,
        "points": points_out,
        "_predicted": predicted,
        "_measured": measured,
        "_de": de,
        "_in_roi": in_roi,
    }


# ---------------------------------------------------------------------------
# 身份（内容寻址）与报告
# ---------------------------------------------------------------------------

def _canon_float(v: float) -> float:
    v = float(v)
    return 0.0 if v == 0.0 else v


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def request_fingerprint(req: AcceptanceRequest) -> dict:
    """验收请求的规范指纹：任务 + 可选修色 + 规范测点序列 + ΔE 上限。

    测点的身份以**解析后的规范 CSV 文本**（行序即复测条目顺序）绑定，
    浮点走最短往返表示，六位截断不会合并不同实测值。
    """
    return {
        "job_id": req.job_id,
        "retouch_id": req.retouch_id,
        "limit_de": _canon_float(req.limit_de),
        "measurements_csv": req.csv_canonical,
    }


def compute_acceptance_id(req: AcceptanceRequest) -> str:
    return hashlib.sha256(
        _canonical_json(request_fingerprint(req)).encode("utf-8")
    ).hexdigest()


def build_report(acceptance_id: str, req: AcceptanceRequest, meta: dict,
                 press_summary: dict, retouch_report: dict | None,
                 result: dict) -> dict:
    """验收报告：绑定任务、可选修色、印刷 ICC 摘要与全部测点结论。"""
    return {
        "acceptance_id": acceptance_id,
        "request": request_fingerprint(req),
        "source_job": {
            "job_id": meta.get("job_id"),
            "image_sha256": meta.get("image_sha256"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "image_mode": meta.get("mode"),
            "source_intent": meta.get("source_intent"),
            "proof_intent": meta.get("proof_intent"),
            "threshold_delta_e": meta.get("threshold"),
        },
        "retouch": (
            None if retouch_report is None
            else {
                "retouch_id": req.retouch_id,
                "region": retouch_report["region"],
                "best_offset_cmyk": retouch_report["best_offset_cmyk"],
                "reachable": True,
            }
        ),
        "profiles": {"press": press_summary},
        "prediction_basis": (
            "选区内测点：核验修色成品绑定后按原分色 CMYK 与修色报告统一整数"
            "偏移重建，经原印刷 ICC、原任务 source→press 意图回算 Lab；"
            "选区外测点：保持原任务 lab_press.npy 逐像素预测值。"
            if retouch_report is not None
            else "全部测点使用原任务 lab_press.npy：分色 CMYK 经原印刷 ICC、"
                 "原任务 source→press 意图回算的逐像素预测 Lab。"
        ),
        "limit_de00": result["limit_de00"],
        "passed": result["passed"],
        "point_count": result["point_count"],
        "over_count": result["over_count"],
        "max_de00": result["max_de00"],
        "mean_de00": result["mean_de00"],
        "over_coordinates": result["over_coordinates"],
        "points": result["points"],
        "artifacts": {
            "report": "report.json",
            "measurements": "measurements.csv",
            "annotated_png": "annotated.png",
        },
        "note": (
            "实测 D50 Lab 与预测 Lab 逐点计算 CIEDE2000；同坐标复测逐条保留，"
            "任一测点 ΔE00 大于上限即整张不合格。本报告不使用原任务值代替"
            "修色选区内的预测；印刷 ICC 缺失/摘要不符、修色不可达或成品绑定"
            "不符、测点越界或 CSV 非法时均拒绝验收。"
        ),
    }


# ---------------------------------------------------------------------------
# 超限坐标标注图
# ---------------------------------------------------------------------------

def render_annotated_png(softproof_png_bytes: bytes, result: dict,
                         retouch_report: dict | None) -> bytes:
    """在原任务软打样 PNG 上标注测点：超限红圈、合格绿圈、复测序号。

    标注为合成可视化，但底图本身是嵌入显示 ICC 的软打样；保存时沿用底图
    嵌入的 ICC 字节，像素（除标注外）不改动。修色选区以橙色矩形勾边。
    """
    try:
        im = Image.open(io.BytesIO(softproof_png_bytes))
        im.load()
    except Exception as exc:
        raise AcceptanceError(f"原任务软打样图无法打开，无法生成标注: {exc}")
    if im.mode != "RGB":
        im = im.convert("RGB")
    embedded = im.info.get("icc_profile")

    draw = ImageDraw.Draw(im)
    w, h = im.size
    r = max(4, min(w, h) // 160)
    lw = max(1, min(w, h) // 400)

    if retouch_report is not None:
        reg = retouch_report["region"]
        draw.rectangle(
            [reg["x"], reg["y"], reg["x"] + reg["width"] - 1,
             reg["y"] + reg["height"] - 1],
            outline=(255, 153, 0), width=max(2, lw),
        )

    for p in result["points"]:
        x, y = p["x"], p["y"]
        color = (230, 0, 0) if not p["within_limit"] else (0, 150, 60)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color,
                     width=max(2, lw))
        if not p["within_limit"]:
            # 超限点叠加十字，便于在密集测点中辨认。
            draw.line([x - r - 2, y, x + r + 2, y], fill=color, width=lw)
            draw.line([x, y - r - 2, x, y + r + 2], fill=color, width=lw)

    buf = io.BytesIO()
    im.save(buf, format="PNG", icc_profile=embedded)
    return buf.getvalue()


def list_retouches_for_job(retouch_dir: Path, job_id: str) -> list[dict]:
    """列出某任务下**可作为验收依据**的修色（报告存在且达标、成品存在）。

    仅做轻量筛选（不做完整绑定核验；完整核验在提交验收时逐字节执行）。
    """
    out: list[dict] = []
    if not retouch_dir.exists():
        return out
    for child in sorted(retouch_dir.iterdir()):
        report_path = child / "report.json"
        if not child.is_dir() or not report_path.is_file():
            continue
        try:
            rep = json.loads(report_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rep, dict):
            continue
        if rep.get("source_job", {}).get("job_id") != job_id:
            continue
        rid = rep.get("retouch_id")
        if rid != child.name:
            continue
        reachable = bool(rep.get("reachable"))
        has_tiff = (child / "retouched.tif").is_file()
        out.append({
            "retouch_id": rid,
            "region": rep.get("region"),
            "best_offset_cmyk": rep.get("best_offset_cmyk"),
            "reachable": reachable,
            "tiff_present": has_tiff,
            "usable": reachable and has_tiff,
        })
    return out
