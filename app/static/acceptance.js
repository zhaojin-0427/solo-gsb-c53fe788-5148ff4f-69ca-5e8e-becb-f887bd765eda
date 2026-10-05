// 印后测色验收视图（原生 JavaScript）
"use strict";

const acState = {
  jobs: [],
  jobSummary: null,
  retouches: [],
  result: null,
};

const ac$ = (id) => document.getElementById(id);

async function acApi(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j && j.detail) msg = j.detail;
    } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

function acShowError(msg) {
  const box = ac$("acError");
  if (!msg) { box.hidden = true; box.textContent = ""; return; }
  box.hidden = false;
  box.textContent = msg;
}

async function acLoadJobs(selectId) {
  acState.jobs = await acApi("/api/jobs");
  const sel = ac$("acJobSelect");
  sel.innerHTML = "";
  if (!acState.jobs.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "（尚无已保存任务，请先在“软打样”页运行一次）";
    sel.appendChild(opt);
    acState.retouches = [];
    acFillRetouches();
    return;
  }
  for (const j of acState.jobs) {
    const opt = document.createElement("option");
    opt.value = j.job_id;
    opt.textContent =
      `${j.job_id.slice(0, 12)}… ${j.width}×${j.height} ${j.image_mode} · ` +
      `${new Date(j.created_at + "Z").toLocaleString()} · ΔE均值 ${j.de_mean?.toFixed(2) ?? "-"}`;
    sel.appendChild(opt);
  }
  if (selectId) sel.value = selectId;
}

async function acSelectJob() {
  const id = ac$("acJobSelect").value;
  acState.jobSummary = acState.jobs.find((j) => j.job_id === id) || null;
  ac$("acResultPanel").hidden = true;
  if (!id) {
    ac$("acJobMeta").textContent = "";
    acState.retouches = [];
    acFillRetouches();
    return;
  }
  const [job, retouches] = await Promise.all([
    acApi(`/api/jobs/${id}`),
    acApi(`/api/jobs/${id}/retouches`),
  ]);
  ac$("acJobMeta").innerHTML =
    `任务 <code>${id.slice(0, 12)}…</code> · ${job.image.width}×` +
    `${job.image.height} · 印刷配置 <b>${job.profiles.press.name}</b> ` +
    `(${job.profiles.press.color_space}) · 意图 ${job.parameters.source_intent_label}`;
  acState.retouches = retouches;
  acFillRetouches();
}

function acFillRetouches() {
  const sel = ac$("acRetouchSelect");
  const keep = sel.value;
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "不绑定修色（按原任务分色预测验收）";
  sel.appendChild(none);
  for (const r of acState.retouches) {
    const opt = document.createElement("option");
    opt.value = r.retouch_id;
    const reg = r.region;
    opt.textContent =
      `修色 ${r.retouch_id.slice(0, 8)}… 选区(${reg.x},${reg.y},` +
      `${reg.width}×${reg.height}) 偏移(${r.best_offset_cmyk?.join(",") ?? "-"})` +
      (r.usable ? "" : "（不可用：未达标或缺成品）");
    if (!r.usable) opt.disabled = true;
    sel.appendChild(opt);
  }
  if (keep && [...sel.options].some((o) => o.value === keep)) sel.value = keep;
  acRenderRetouchMeta();
}

function acRenderRetouchMeta() {
  const rid = ac$("acRetouchSelect").value;
  const box = ac$("acRetouchMeta");
  if (!rid) {
    box.textContent = "不绑定修色：全部测点使用原任务 lab_press 逐像素预测。";
    return;
  }
  const r = acState.retouches.find((x) => x.retouch_id === rid);
  if (!r) { box.textContent = ""; return; }
  box.innerHTML =
    `绑定达标修色成品 <code>${rid.slice(0, 12)}…</code>：选区内测点按原分色 + ` +
    `统一偏移 (${r.best_offset_cmyk.join(", ")}) 经原印刷 ICC 回算；` +
    `选区外测点保持原任务预测。提交时将独立复核成品绑定。`;
}

// ---- 运行验收 -------------------------------------------------------------
ac$("acRefreshJobs").addEventListener("click", () =>
  acLoadJobs(ac$("acJobSelect").value).catch((e) => acShowError(e.message))
);
ac$("acJobSelect").addEventListener("change", () =>
  acSelectJob().catch((e) => acShowError(e.message))
);
ac$("acRetouchSelect").addEventListener("change", acRenderRetouchMeta);

ac$("acRunBtn").addEventListener("click", async () => {
  acShowError("");
  const jobId = ac$("acJobSelect").value;
  if (!jobId) return acShowError("请先选择一个已保存任务");
  const file = ac$("acCsvFile").files[0];
  if (!file) return acShowError("请先选择测点 CSV 文件");
  const limit = parseFloat(ac$("acLimit").value);
  if (!Number.isFinite(limit) || limit < 0) {
    return acShowError("ΔE00 上限必须是非负有限数值");
  }
  const fd = new FormData();
  fd.append("job_id", jobId);
  fd.append("limit_de", String(limit));
  fd.append("file", file);
  const rid = ac$("acRetouchSelect").value;
  if (rid) fd.append("retouch_id", rid);

  ac$("acRunBtn").disabled = true;
  ac$("acBusy").hidden = false;
  try {
    acState.result = await acApi("/api/acceptance", { method: "POST", body: fd });
    acRenderResult();
  } catch (err) {
    ac$("acResultPanel").hidden = true;
    acShowError(err.message);
  } finally {
    ac$("acRunBtn").disabled = false;
    ac$("acBusy").hidden = true;
  }
});

function acLabText(v) {
  return Array.isArray(v) ? `(${v.map((x) => x.toFixed(2)).join(", ")})` : "—";
}

function acRenderResult() {
  const r = acState.result;
  ac$("acResultPanel").hidden = false;
  const status = ac$("acStatus");
  if (r.passed) {
    status.className = "rt-status reachable";
    status.textContent =
      `验收合格：${r.point_count} 条测点全部满足 ΔE00 ≤ ${r.limit_de00}，` +
      `最大 ΔE00=${r.max_de00}。`;
  } else {
    status.className = "rt-status unreachable";
    status.textContent =
      `验收不合格：${r.point_count} 条测点中有 ${r.over_count} 条 ` +
      `ΔE00 大于上限 ${r.limit_de00}，整张判废；最大 ΔE00=${r.max_de00}。`;
  }

  ac$("acStats").innerHTML =
    `<span>测点数（含复测）<b>${r.point_count}</b></span>` +
    `<span>超限点数 <b>${r.over_count}</b></span>` +
    `<span>ΔE00 均值 <b>${r.mean_de00}</b></span>` +
    `<span>ΔE00 最大 <b>${r.max_de00}</b></span>` +
    `<span>上限 <b>${r.limit_de00}</b></span>` +
    (r.retouch_id
      ? `<span>绑定修色 <code>${r.retouch_id.slice(0, 12)}…</code></span>`
      : `<span>无修色（原任务预测）</span>`);

  ac$("acAnnotated").src = `${r.annotated_url}?cache=${r.acceptance_id}`;

  const over = r.points.filter((p) => !p.within_limit);
  const overBox = ac$("acOverList");
  if (!over.length) {
    overBox.innerHTML = "<h3>超限坐标</h3><div class=\"kv\">无超限测点。</div>";
  } else {
    const rows = over.map((p) =>
      `<tr><td>${p.index}</td><td>(${p.x}, ${p.y})</td>` +
      `<td>${acLabText(p.measured_lab)}</td><td>${acLabText(p.predicted_lab)}</td>` +
      `<td><b>${p.de00}</b></td><td>${p.source === "retouched_region" ? "修色选区" : "原任务预测"}</td></tr>`
    ).join("");
    overBox.innerHTML =
      `<h3>超限坐标（${over.length} 条，同坐标复测逐条列出）</h3>` +
      `<table class="ac-table"><thead><tr><th>#</th><th>坐标(x,y)</th>` +
      `<th>实测 Lab</th><th>预测 Lab</th><th>ΔE00</th><th>预测来源</th></tr></thead>` +
      `<tbody>${rows}</tbody></table>`;
  }

  ac$("acExpReport").href = r.report_url;
  ac$("acExpReport").download = `acceptance-${r.acceptance_id.slice(0, 12)}.json`;
  ac$("acExpAnnotated").href = r.annotated_url;
  ac$("acExpAnnotated").download = `acceptance-${r.acceptance_id.slice(0, 12)}.png`;
  ac$("acExpCsv").href = `/api/acceptance/${r.acceptance_id}/measurements.csv`;
  ac$("acExpCsv").download = `measurements-${r.acceptance_id.slice(0, 12)}.csv`;
  ac$("acReportJson").textContent = JSON.stringify(r.report, null, 2);
}

// ---- 视图切换钩子（app.js 的导航处理调用） -------------------------------
async function refreshAcceptanceView() {
  try {
    if (!acState.jobs.length) await acLoadJobs();
    if (!ac$("acJobSelect").value && acState.jobs.length) await acSelectJob();
  } catch (e) {
    acShowError("加载已保存任务失败: " + e.message);
  }
}
