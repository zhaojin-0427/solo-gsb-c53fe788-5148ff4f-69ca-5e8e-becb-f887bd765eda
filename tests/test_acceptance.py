"""印后测色验收端到端测试。

覆盖：CSV 严格校验、逐点 ΔE00 与整张判定、同坐标复测逐条保留、按原任务
预测 / 绑定达标修色（原分色+偏移重建回算，选区外保持原预测）、超限坐标
标注、报告绑定与确定性复用，以及修色不可达 / 报告与成品绑定不符 / ICC
缺失与摘要不符 / 测点越界等拒绝路径。
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import lcms_native
from app.main import app


@pytest.fixture
def client(tmp_path):
    data_dir = tmp_path / "acceptance-data"
    os.environ["SPD_DATA_DIR"] = str(data_dir)
    from app import config
    config._state = None
    config.DATA_DIR = data_dir
    config.PROFILE_DIR = data_dir / "profiles"
    config.JOB_DIR = data_dir / "jobs"
    config.RETOUCH_DIR = data_dir / "retouch"
    config.ACCEPTANCE_DIR = data_dir / "acceptance"
    config.RELEASE_DIR = data_dir / "release"
    config.DB_PATH = data_dir / "softproof.db"
    from app import main
    main.JOB_DIR = data_dir / "jobs"
    main.RETOUCH_DIR = data_dir / "retouch"
    main.ACCEPTANCE_DIR = data_dir / "acceptance"
    main.DB_PATH = data_dir / "softproof.db"
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ids(client):
    profs = client.get("/api/profiles").json()
    return {
        "srgb": next(p["profile_id"] for p in profs["source"]
                     if p["color_space"] == "RGB"),
        "cmyk": next(p["profile_id"] for p in profs["press"]),
        "display": next(p["profile_id"] for p in profs["display"]),
    }


def _solid_png(rgb=(210, 90, 60), size=48):
    arr = np.zeros((size, size, 3), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _gradient_png(size=48):
    yy, xx = np.mgrid[0:size, 0:size]
    arr = np.stack([(xx * 5) % 256, (yy * 5) % 256,
                    ((xx + yy) * 3) % 256], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _run_job(client, ids, png=_solid_png()):
    r = client.post(
        "/api/jobs",
        files={"image": ("img.png", png, "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


def _accept(client, job_id, csv_bytes, limit="3.0", retouch_id=None):
    data = {"job_id": job_id, "limit_de": limit}
    if retouch_id is not None:
        data["retouch_id"] = retouch_id
    return client.post(
        "/api/acceptance",
        files={"file": ("points.csv", csv_bytes, "text/csv")},
        data=data,
    )


def _lab_at(client, ids, job_id, coords):
    """直接读原任务 lab_press.npy 的指定坐标预测 Lab。"""
    from app import main
    lab = np.load(main.JOB_DIR / job_id / "lab_press.npy", mmap_mode="r")
    return np.asarray([[float(v) for v in lab[y, x]] for x, y in coords])


def _retouch_reachable(client, job_id, rect=(4, 4, 16, 16), offset=(3, 0, 0, 0)):
    """以“原选区 CMYK 统一加 offset 后”的平均 Lab 为目标，保证可达。"""
    from app import main
    cmyk = np.load(main.JOB_DIR / job_id / "cmyk.npy")
    x, y, w, h = rect
    meta = json.loads((main.JOB_DIR / job_id / "meta.json").read_text("utf-8"))
    press = client.app.state.state.pm.load_bytes(meta["press_profile"])
    roi = cmyk[y:y + h, x:x + w].reshape(-1, 4).astype(np.int32)
    roi = np.clip(roi + np.array(offset, dtype=np.int32), 0, 255).astype(np.uint8)
    target = lcms_native.pixels_to_lab(
        np.ascontiguousarray(roi), "CMYK", press, "relative_colorimetric"
    ).mean(axis=0)
    body = {
        "job_id": job_id, "x": x, "y": y, "width": w, "height": h,
        "target_lab": [float(v) for v in target],
        "tolerance_de": 0.5, "max_shift": 8, "tac_limit": 1020,
    }
    r = client.post("/api/retouch", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reachable"], d
    return d


# --------------------------------------------------------------------------
# CSV 校验
# --------------------------------------------------------------------------

def test_health(client):
    assert client.get("/api/health").status_code == 200


def test_empty_csv_rejected(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"], b"")
    assert r.status_code == 422
    assert "CSV" in r.json()["detail"]


def test_bad_header_rejected(client, ids):
    job = _run_job(client, ids)
    for text in (
        "xx,yy,L,a,b\n1,2,50,0,0\n",
        "x,y,L,a\n1,2,50,0\n",
        "x,y,L,a,b,extra\n1,2,50,0,0,1\n",
        "1,2,50,0,0\n",
    ):
        r = _accept(client, job["job_id"], _csv(text))
        assert r.status_code == 422, text
    # 表头列名大小写不敏感（x,y,L,a,b 任意大小写均可）。
    r = _accept(client, job["job_id"], _csv("X,Y,L,A,B\n0,0,50,0,0\n"))
    assert r.status_code == 200, r.text


def test_non_integer_xy_rejected(client, ids):
    job = _run_job(client, ids)
    for text in (
        "x,y,L,a,b\n1.0,2,50,0,0\n",
        "x,y,L,a,b\n-1,2,50,0,0\n",
        "x,y,L,a,b\n1,,50,0,0\n",
        "x,y,L,a,b\n0x1,2,50,0,0\n",
        "x,y,L,a,b\n１,2,50,0,0\n",
    ):
        r = _accept(client, job["job_id"], _csv(text))
        assert r.status_code == 422, text


def test_lab_ranges_and_finiteness_rejected(client, ids):
    job = _run_job(client, ids)
    for text in (
        "x,y,L,a,b\n1,2,100.1,0,0\n",
        "x,y,L,a,b\n1,2,-0.1,0,0\n",
        "x,y,L,a,b\n1,2,50,128,0\n",
        "x,y,L,a,b\n1,2,50,0,-129\n",
        "x,y,L,a,b\n1,2,NaN,0,0\n",
        "x,y,L,a,b\n1,2,50,Infinity,0\n",
        "x,y,L,a,b\n1,2,,0,0\n",
    ):
        r = _accept(client, job["job_id"], _csv(text))
        assert r.status_code == 422, text


def test_wrong_column_count_and_blank_line_rejected(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"], _csv("x,y,L,a,b\n1,2,50,0\n"))
    assert r.status_code == 422
    r = _accept(client, job["job_id"], _csv("x,y,L,a,b\n1,2,50,0,0\n\n"))
    assert r.status_code == 422


def test_header_only_rejected(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"], _csv("x,y,L,a,b\n"))
    assert r.status_code == 422


# --------------------------------------------------------------------------
# 逐点验收：预测 Lab 取自原任务 lab_press，ΔE00 与判定
# --------------------------------------------------------------------------

def test_pass_when_measured_equals_predicted(client, ids):
    job = _run_job(client, ids)
    lab = _lab_at(client, ids, job["job_id"], [(0, 0), (10, 10), (47, 47)])
    text = "x,y,L,a,b\n" + "".join(
        f"{x},{y},{L:.10f},{a:.10f},{b:.10f}\n"
        for (x, y), (L, a, b) in zip(
            [(0, 0), (10, 10), (47, 47)], lab)
    )
    r = _accept(client, job["job_id"], _csv(text), limit="0")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["passed"] is True
    assert d["over_count"] == 0
    assert d["max_de00"] < 1e-6
    assert d["point_count"] == 3


def test_whole_sheet_fails_if_any_point_over_limit(client, ids):
    job = _run_job(client, ids)
    lab = _lab_at(client, ids, job["job_id"], [(0, 0), (10, 10)])
    text = "x,y,L,a,b\n"
    text += f"0,0,{lab[0][0]:.10f},{lab[0][1]:.10f},{lab[0][2]:.10f}\n"
    text += f"10,10,{lab[1][0] + 8.0:.10f},{lab[1][1]:.10f},{lab[1][2]:.10f}\n"
    r = _accept(client, job["job_id"], _csv(text), limit="3")
    assert r.status_code == 200
    d = r.json()
    assert d["passed"] is False
    assert d["over_count"] == 1
    assert d["over_coordinates"] == [[10, 10]]
    assert d["points"][0]["within_limit"] is True
    assert d["points"][1]["within_limit"] is False
    assert d["points"][1]["de00"] > 3.0


def test_de_equal_to_limit_passes(client, ids):
    """ΔE00 恰好等于上限：不大于上限，判合格（用实测 ΔE 反推上限，避免
    依赖 ΔE 与 L 微扰量严格相等的假设）。"""
    job = _run_job(client, ids)
    lab = _lab_at(client, ids, job["job_id"], [(5, 5)])
    L, a, b = lab[0]
    text = f"x,y,L,a,b\n5,5,{L + 2.5:.10f},{a:.10f},{b:.10f}\n"
    probe = _accept(client, job["job_id"], _csv(text), limit="0").json()
    de = probe["points"][0]["de00"]
    assert de > 0
    # 上限取报告所示 ΔE 值（4 位小数，对用户而言就是“等于实测 ΔE00”），
    # 允许半个报告量化单位：语义上“不大于上限即合格”，实现只保留 1e-9 的
    # 严格数值容差，因此这里再补 5e-5（报告 ΔE 的量化步长）后必须合格。
    d = _accept(client, job["job_id"], _csv(text),
                limit=repr(float(de) + 5e-5)).json()
    assert d["passed"] is True
    # 上限显著低于实测 ΔE 则不合格。
    d2 = _accept(client, job["job_id"], _csv(text),
                 limit=repr(float(de) - 0.1)).json()
    assert d2["passed"] is False


def test_duplicate_same_coordinate_rows_kept_separately(client, ids):
    job = _run_job(client, ids)
    lab = _lab_at(client, ids, job["job_id"], [(7, 7)])
    L, a, b = lab[0]
    text = (
        "x,y,L,a,b\n"
        f"7,7,{L:.6f},{a:.6f},{b:.6f}\n"
        f"7,7,{L + 9:.6f},{a:.6f},{b:.6f}\n"
    )
    d = _accept(client, job["job_id"], _csv(text), limit="3").json()
    assert d["point_count"] == 2
    assert d["points"][0]["x"] == 7 and d["points"][1]["x"] == 7
    assert d["passed"] is False
    assert d["over_coordinates"] == [[7, 7]]


def test_out_of_bounds_coordinates_rejected(client, ids):
    job = _run_job(client, ids)
    for text in (
        "x,y,L,a,b\n48,0,50,0,0\n",
        "x,y,L,a,b\n0,48,50,0,0\n",
        "x,y,L,a,b\n100,100,50,0,0\n",
    ):
        r = _accept(client, job["job_id"], _csv(text))
        assert r.status_code == 422
        assert "越界" in r.json()["detail"]


def test_unknown_job_404(client, ids):
    r = _accept(client, "0" * 64, _csv("x,y,L,a,b\n0,0,50,0,0\n"))
    assert r.status_code == 404


def test_negative_limit_rejected(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"],
                _csv("x,y,L,a,b\n0,0,50,0,0\n"), limit="-0.1")
    assert r.status_code == 422


# --------------------------------------------------------------------------
# 报告 / 标注图 / 产物
# --------------------------------------------------------------------------

def test_report_binds_job_press_icc_and_points(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"],
                _csv("x,y,L,a,b\n0,0,50,0,0\n1,1,60,2,-3\n"), limit="3")
    assert r.status_code == 200
    d = r.json()
    aid = d["acceptance_id"]
    rep = client.get(f"/api/acceptance/{aid}/report").json()
    assert rep["source_job"]["job_id"] == job["job_id"]
    assert rep["source_job"]["image_sha256"] == job["report"]["image"]["sha256"]
    assert rep["profiles"]["press"]["sha256"] == \
        job["report"]["profiles"]["press"]["sha256"]
    assert rep["profiles"]["press"]["color_space"] == "CMYK"
    assert rep["retouch"] is None
    assert rep["limit_de00"] == 3.0
    assert len(rep["points"]) == 2
    assert rep["points"][0]["source"] == "job_lab_press"
    assert rep["artifacts"]["annotated_png"] == "annotated.png"

    png = client.get(f"/api/acceptance/{aid}/annotated.png")
    assert png.status_code == 200
    im = Image.open(io.BytesIO(png.content))
    assert im.size == (48, 48)
    # 底图为嵌入显示 ICC 的原软打样。
    display = client.app.state.state.pm.load_bytes(ids["display"])
    assert im.info.get("icc_profile") == display

    csv_resp = client.get(f"/api/acceptance/{aid}/measurements.csv")
    assert csv_resp.status_code == 200
    assert csv_resp.content.startswith(b"x,y,L,a,b\n")


def test_deterministic_acceptance_id_and_reuse(client, ids):
    job = _run_job(client, ids)
    text = "x,y,L,a,b\n0,0,50,0,0\n"
    d1 = _accept(client, job["job_id"], _csv(text)).json()
    d2 = _accept(client, job["job_id"], _csv(text)).json()
    assert d1["acceptance_id"] == d2["acceptance_id"]
    assert d1 == d2
    # 不同测点 / 上限得到不同身份。
    d3 = _accept(client, job["job_id"],
                 _csv("x,y,L,a,b\n0,0,50.1,0,0\n")).json()
    d4 = _accept(client, job["job_id"], _csv(text), limit="4").json()
    assert d3["acceptance_id"] != d1["acceptance_id"]
    assert d4["acceptance_id"] != d1["acceptance_id"]


def test_csv_float_precision_not_merged(client, ids):
    job = _run_job(client, ids)
    a = _accept(client, job["job_id"],
                _csv("x,y,L,a,b\n0,0,50.0000001,0,0\n")).json()
    b = _accept(client, job["job_id"],
                _csv("x,y,L,a,b\n0,0,50.0000002,0,0\n")).json()
    assert a["acceptance_id"] != b["acceptance_id"]


def test_annotated_image_marks_over_points(client, ids):
    job = _run_job(client, ids)
    lab = _lab_at(client, ids, job["job_id"], [(0, 0)])
    L, a_, b_ = lab[0]
    d = _accept(
        client, job["job_id"],
        _csv(f"x,y,L,a,b\n0,0,{L + 9:.6f},{a_:.6f},{b_:.6f}\n"),
        limit="3",
    ).json()
    assert d["passed"] is False
    png = client.get(d["annotated_url"]).content
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    # 标注红圈附近应出现高红低绿像素（标注确实落在图上）。
    patch = im[0:12, 0:12].reshape(-1, 3).astype(np.int32)
    reddish = ((patch[:, 0] > 180) & (patch[:, 1] < 90) & (patch[:, 2] < 90))
    assert reddish.any()


# --------------------------------------------------------------------------
# 绑定达标修色成品
# --------------------------------------------------------------------------

def test_retouch_listing_only_usable(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    lst = client.get(f"/api/jobs/{job['job_id']}/retouches").json()
    assert lst == []
    rd = _retouch_reachable(client, job["job_id"])
    lst = client.get(f"/api/jobs/{job['job_id']}/retouches").json()
    assert len(lst) == 1
    assert lst[0]["retouch_id"] == rd["retouch_id"]
    assert lst[0]["usable"] is True


def test_acceptance_with_retouch_rebuilds_region_predictions(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rect = (4, 4, 16, 16)
    rd = _retouch_reachable(client, job["job_id"], rect=rect, offset=(3, 0, 0, 0))
    from app import main
    cmyk = np.load(main.JOB_DIR / job["job_id"] / "cmyk.npy")
    meta = json.loads((main.JOB_DIR / job["job_id"] / "meta.json").read_text())
    press = client.app.state.state.pm.load_bytes(meta["press_profile"])

    # 选区内点：以“偏移后真实回算 Lab”为实测 -> 合格；
    # 选区外点：以原任务 lab_press 为实测。
    inside = (6, 6)
    outside = (30, 30)
    off = np.array(rd["best_offset_cmyk"], dtype=np.int32)
    shifted = np.clip(
        cmyk[inside[1], inside[0]].astype(np.int32) + off, 0, 255
    ).astype(np.uint8)
    lab_in = lcms_native.pixels_to_lab(
        shifted[None, :], "CMYK", press, "relative_colorimetric"
    )[0]
    lab_out = np.load(main.JOB_DIR / job["job_id"] / "lab_press.npy"
                      )[outside[1], outside[0]]
    text = "x,y,L,a,b\n"
    text += f"{inside[0]},{inside[1]},{lab_in[0]:.10f},{lab_in[1]:.10f},{lab_in[2]:.10f}\n"
    text += f"{outside[0]},{outside[1]},{lab_out[0]:.10f},{lab_out[1]:.10f},{lab_out[2]:.10f}\n"
    r = _accept(client, job["job_id"], _csv(text), limit="0",
                retouch_id=rd["retouch_id"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["passed"] is True, d
    assert d["points"][0]["source"] == "retouched_region"
    assert d["points"][1]["source"] == "job_lab_press"
    assert max(p["de00"] for p in d["points"]) < 1e-6
    rep = client.get(d["report_url"]).json()
    assert rep["retouch"]["retouch_id"] == rd["retouch_id"]
    assert rep["retouch"]["region"] == rd["rect"]
    assert rep["retouch"]["best_offset_cmyk"] == rd["best_offset_cmyk"]

    # 反例：绑定修色但选区内仍提交“原任务预测 Lab”作为实测——若实际偏移
    # 非零，它不等于修色后回算值，应判超限（证明没有用原任务值代替）。
    lab_in_orig = np.load(
        main.JOB_DIR / job["job_id"] / "lab_press.npy"
    )[inside[1], inside[0]]
    text2 = "x,y,L,a,b\n"
    text2 += (f"{inside[0]},{inside[1]},"
              f"{lab_in_orig[0]:.10f},{lab_in_orig[1]:.10f},{lab_in_orig[2]:.10f}\n")
    probe = _accept(client, job["job_id"], _csv(text2), limit="0",
                    retouch_id=rd["retouch_id"]).json()
    if any(int(v) != 0 for v in rd["best_offset_cmyk"]):
        assert probe["passed"] is False
        assert probe["points"][0]["de00"] > 0
    else:
        assert probe["passed"] is True


def test_outside_region_keeps_original_prediction_with_retouch(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rd = _retouch_reachable(client, job["job_id"], rect=(4, 4, 16, 16),
                            offset=(2, 1, 0, 0))
    from app import main
    lab_press = np.load(main.JOB_DIR / job["job_id"] / "lab_press.npy")
    x, y = 40, 40  # 选区外
    L, a, b = lab_press[y, x]
    text = f"x,y,L,a,b\n{x},{y},{L:.10f},{a:.10f},{b:.10f}\n"
    d = _accept(client, job["job_id"], _csv(text), limit="0",
                retouch_id=rd["retouch_id"]).json()
    assert d["passed"] is True
    assert d["points"][0]["source"] == "job_lab_press"


def test_unreachable_retouch_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    body = {
        "job_id": job["job_id"], "x": 4, "y": 4, "width": 16, "height": 16,
        "target_lab": [100.0, 127.0, 127.0],
        "tolerance_de": 0.0, "max_shift": 0, "tac_limit": 1020,
    }
    du = client.post("/api/retouch", json=body).json()
    assert du["reachable"] is False
    r = _accept(client, job["job_id"],
                _csv("x,y,L,a,b\n5,5,50,0,0\n"), retouch_id=du["retouch_id"])
    assert r.status_code == 422
    assert "不可达" in r.json()["detail"]


def test_unknown_retouch_404(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"],
                _csv("x,y,L,a,b\n0,0,50,0,0\n"), retouch_id="0" * 64)
    assert r.status_code == 404


def test_retouch_from_other_job_rejected(client, ids):
    job1 = _run_job(client, ids, png=_gradient_png())
    job2 = _run_job(client, ids,
                    png=_gradient_png(size=40) if False else _solid_png((1, 2, 3)))
    rd = _retouch_reachable(client, job1["job_id"])
    r = _accept(client, job2["job_id"],
                _csv("x,y,L,a,b\n0,0,50,0,0\n"), retouch_id=rd["retouch_id"])
    assert r.status_code == 422
    assert "任务不符" in r.json()["detail"]


def test_tampered_retouch_tiff_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rd = _retouch_reachable(client, job["job_id"], rect=(4, 4, 16, 16))
    from app import main
    tif = main.RETOUCH_DIR / rd["retouch_id"] / "retouched.tif"
    blob = tif.read_bytes()
    arr = np.asarray(Image.open(io.BytesIO(blob))).copy()
    v = int(arr[0, 0, 0])  # (0,0) 在选区外
    arr[0, 0, 0] = v + 7 if v < 240 else v - 7
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    buf = io.BytesIO()
    Image.fromarray(arr, "CMYK").save(
        buf, format="TIFF", compression="raw", icc_profile=press)
    tif.write_bytes(buf.getvalue())
    try:
        r = _accept(client, job["job_id"],
                    _csv("x,y,L,a,b\n0,0,50,0,0\n"),
                    retouch_id=rd["retouch_id"])
        assert r.status_code == 422
        assert "选区外" in r.json()["detail"]
    finally:
        tif.write_bytes(blob)


def test_tampered_press_digest_in_retouch_report_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rd = _retouch_reachable(client, job["job_id"])
    from app import main
    rp = main.RETOUCH_DIR / rd["retouch_id"] / "report.json"
    rep = json.loads(rp.read_text("utf-8"))
    rep["profiles"]["press"]["sha256"] = "f" * 64
    rp.write_text(json.dumps(rep), "utf-8")
    try:
        r = _accept(client, job["job_id"],
                    _csv("x,y,L,a,b\n5,5,50,0,0\n"),
                    retouch_id=rd["retouch_id"])
        assert r.status_code == 422
    finally:
        # 修色端点自身的缓存核验会在下次调用时修复；测试间数据目录独立，无需恢复。
        pass


def test_missing_press_icc_rejected(client, ids):
    job = _run_job(client, ids)
    state = client.app.state.state
    target = next(p for role, p in state.pm._iter_files()
                  if role in ("press", "builtin") and p.stem == ids["cmyk"])
    backup = target.read_bytes()
    target.unlink()
    try:
        r = _accept(client, job["job_id"],
                    _csv("x,y,L,a,b\n0,0,50,0,0\n"))
        assert r.status_code == 422
        assert "印刷 ICC" in r.json()["detail"]
    finally:
        target.write_bytes(backup)


def test_press_icc_digest_mismatch_rejected(client, ids):
    job = _run_job(client, ids)
    state = client.app.state.state
    target = next(p for role, p in state.pm._iter_files()
                  if p.stem == ids["cmyk"])
    other = lcms_native.build_demo_cmyk_profile_bytes(ink_scale=0.9)
    original = target.read_bytes()
    assert other != original
    target.write_bytes(other)
    try:
        r = _accept(client, job["job_id"],
                    _csv("x,y,L,a,b\n0,0,50,0,0\n"))
        assert r.status_code == 422
        assert "摘要不符" in r.json()["detail"]
    finally:
        target.write_bytes(original)
