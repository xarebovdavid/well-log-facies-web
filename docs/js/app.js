// Главный сценарий сайта: загрузка файла → расчёт → планшет, интервалы, сравнение с керном.
import { FACIES, FACIES_IDS, FEATURES, FIELD_HELP, faciesLabel, isNear } from "./facies.js";
import { loadModel, predict } from "./model.js";
import { FIELDS, REQUIRED, buildDataset, guessMapping, missingFields, parseCSV, readFile, toCSV } from "./parse.js";
import { createPlot, runs, tooltipHTML } from "./plot.js";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const pct = (v) => Math.round(v * 100) + "%";
const num = (v, digits = 1) => v.toLocaleString("ru-RU", { maximumFractionDigits: digits, useGrouping: false });
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  source: null, // { table, fileName, mapping, logRes } — то, что сейчас на экране
  wells: [], hasTruth: false, current: 0,
  threshold: 0.5, zoom: 1, filter: "all", plot: null,
};

const ready = Promise.all([
  loadModel("model/model.json"),
  fetch("model/info.json").then((r) => r.json()),
]).then(([model, info]) => { state.model = model; state.info = info; renderModelView(info); return model; });
ready.catch(() => toast("Не удалось загрузить модель. Обновите страницу.", "error"));

// ---------- загрузка данных ----------
async function openFile(file) {
  try {
    setBusy(true);
    const table = await readFile(file);
    startWith(table, file.name);
  } catch (e) {
    toast(`Не удалось прочитать «${file.name}»: ${e.message}`, "error");
  } finally { setBusy(false); }
}

async function openExample() {
  try {
    setBusy(true);
    const text = await (await fetch("data/example.csv")).text();
    startWith(parseCSV(text), "Пример: две скважины, которых модель не видела при обучении");
  } catch { toast("Не удалось загрузить пример.", "error"); } finally { setBusy(false); }
}

function startWith(table, fileName) {
  const guess = guessMapping(table.columns);
  const source = { table, fileName, mapping: guess.mapping, logRes: guess.logRes };
  if (missingFields(guess.mapping).length) openMapping(source);
  else compute(source);
}

async function compute(source) {
  try {
    const model = await ready;
    const fallbackWell = source.fileName.replace(/\.[^.]+$/, "") || "Скважина";
    const data = buildDataset(source.table, source.mapping, { logRes: source.logRes, fallbackWell });
    state.wells = data.wells.map((w) => {
      const result = predict(model, w.x);
      const intervals = runs(result.pred, w.depth, w.step).map((r) => {
        let sum = 0;
        for (let i = r.from; i <= r.to; i++) sum += result.conf[i];
        return { ...r, conf: sum / (r.to - r.from + 1) };
      });
      return { ...w, ...result, intervals, edits: new Map(), final: null };
    });
    Object.assign(state, { source, hasTruth: data.hasTruth, current: 0, filter: "all", zoom: 1 });

    $("#landing").hidden = true;
    $("#workspace").hidden = false;
    $("#source").textContent = source.fileName;
    $("#notes").innerHTML = data.notes.map((n) => `<p class="note">${esc(n)}</p>`).join("");
    showView("analysis");
    renderWells();
    renderWell(true);
    window.scrollTo({ top: 0 });
  } catch (e) {
    toast(e.message, "error");
  }
}

function setBusy(on) { document.body.classList.toggle("busy", on); }

// ---------- скважина ----------
const well = () => state.wells[state.current];

function renderWells() {
  const el = $("#wells");
  el.hidden = state.wells.length < 2;
  el.innerHTML = state.wells.map((w, i) =>
    `<button class="chip${i === state.current ? " active" : ""}" role="tab" data-i="${i}">${esc(w.name)}<small>${w.depth.length}</small></button>`).join("");
}

function renderWell(animate) {
  renderMetrics(animate);
  renderPlot(animate);
  renderComposition();
  renderIntervals();
  renderCompare();
}

function doubtfulShare(w) {
  let low = 0;
  for (const c of w.conf) if (c < state.threshold) low++;
  return low / w.conf.length;
}

function renderMetrics(animate) {
  const w = well(), n = w.depth.length;
  const cards = [
    { key: "depth", label: "Глубины, м", text: `${num(w.depth[0], 0)}–${num(w.depth[n - 1], 0)}` },
    { key: "points", label: "Точек", value: n, format: (v) => num(Math.round(v), 0) },
    { key: "conf", label: "Средняя уверенность", value: w.conf.reduce((a, b) => a + b, 0) / n, format: pct },
    { key: "doubt", label: "Сомнительных", value: doubtfulShare(w), format: pct, tone: "warn", hint: "Доля глубин с уверенностью ниже порога" },
  ];
  if (w.truth) {
    let exact = 0, near = 0;
    for (let i = 0; i < n; i++) { if (w.truth[i] === w.pred[i]) exact++; if (isNear(w.truth[i], w.pred[i])) near++; }
    cards.push({ key: "match", label: "Совпадение с керном", value: exact / n, format: pct, tone: "good", sub: `с соседними фациями: ${pct(near / n)}` });
  }
  $("#metrics").innerHTML = cards.map(metricCard).join("");
  cards.forEach((c) => c.value !== undefined && countUp($(`#metrics [data-key="${c.key}"] .metric-value`), c.value, c.format, animate));
}

const metricCard = (c, i) => `
  <div class="metric ${c.tone ?? ""}" data-key="${c.key}" style="--i:${i}" ${c.hint ? `title="${c.hint}"` : ""}>
    <div class="metric-label">${c.label}</div>
    <div class="metric-value">${c.text ?? ""}</div>
    ${c.sub ? `<div class="metric-sub">${c.sub}</div>` : ""}
  </div>`;

function countUp(el, to, format, animate = true) {
  if (!animate || reducedMotion) { el.textContent = format(to); return; }
  const start = performance.now(), duration = 700;
  const tick = (now) => {
    const t = Math.min((now - start) / duration, 1);
    el.textContent = format(to * (1 - Math.pow(1 - t, 3)));
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderPlot(animate) {
  const w = well();
  const span = w.depth[w.depth.length - 1] - w.depth[0] + w.step;
  const base = Math.min(Math.max(w.depth.length * 2.6, 640), 3600) / span;
  state.plot = createPlot($("#plot"), w, {
    threshold: state.threshold,
    pxPerMeter: base * state.zoom,
    animate: animate && !reducedMotion,
    onHover: showTip,
  });
}

/** Перерисовать планшет, сохранив место, на которое смотрит пользователь. */
function redrawPlotKeepingPlace() {
  const body = $("#plot .plot-body");
  const before = body.getBoundingClientRect();
  const anchor = (innerHeight / 2 - before.top) / before.height; // доля высоты планшета в центре экрана
  renderPlot(false);
  if (anchor > 0 && anchor < 1) {
    const after = $("#plot .plot-body").getBoundingClientRect();
    window.scrollBy(0, after.top + after.height * anchor - innerHeight / 2);
  }
}

const tip = $("#tip");
let tipIndex = -1;
function showTip(i, e) {
  if (i < 0) { tip.classList.remove("show"); tipIndex = -1; return; }
  if (i !== tipIndex) { tip.innerHTML = tooltipHTML(well(), i, state.threshold); tipIndex = i; }
  const pad = 18, { offsetWidth: tw, offsetHeight: th } = tip;
  const left = e.clientX + pad + tw > innerWidth ? e.clientX - pad - tw : e.clientX + pad;
  const top = Math.min(Math.max(e.clientY - th / 2, 8), innerHeight - th - 8);
  tip.style.transform = `translate(${Math.max(left, 8)}px, ${top}px)`;
  tip.classList.add("show");
}

function renderComposition() {
  const w = well(), counts = Object.fromEntries(FACIES_IDS.map((k) => [k, 0]));
  for (const f of w.final ?? w.pred) counts[f]++;
  const max = Math.max(...Object.values(counts));
  $("#composition").innerHTML = FACIES_IDS.map((k) => `
    <li class="${counts[k] ? "" : "absent"}" title="${esc(faciesLabel(k))}">
      <i style="background:${FACIES[k].color}"></i>
      <span class="comp-name"><b>${FACIES[k].code}</b> ${FACIES[k].name}</span>
      <span class="comp-value">${pct(counts[k] / w.pred.length)}</span>
      <span class="comp-bar"><i style="width:${(counts[k] / max) * 100}%;background:${FACIES[k].color}"></i></span>
    </li>`).join("");
}

// ---------- интервалы ----------
function renderIntervals() {
  const w = well();
  const isDoubt = (r) => r.conf < state.threshold;
  const isChanged = (r) => w.edits.has(r.from);
  const filters = [
    ["all", "Все", w.intervals.length],
    ["doubt", "Сомнительные", w.intervals.filter(isDoubt).length],
    ["changed", "Изменённые", w.intervals.filter(isChanged).length],
  ];
  $("#interval-filter").innerHTML = filters.map(([key, label, count]) =>
    `<button class="chip${state.filter === key ? " active" : ""}" data-filter="${key}">${label}<small>${count}</small></button>`).join("");

  const rows = w.intervals.filter((r) => state.filter === "all" || (state.filter === "doubt" ? isDoubt(r) : isChanged(r)));
  const options = (selected) => FACIES_IDS.map((k) =>
    `<option value="${k}"${k === selected ? " selected" : ""}>${esc(faciesLabel(k))}</option>`).join("");
  $("#intervals").innerHTML = rows.map((r) => {
    const top = w.depth[r.from], bottom = w.depth[r.to] + w.step;
    const f = FACIES[r.value];
    return `
      <tr data-from="${r.from}" data-to="${r.to}" class="${isDoubt(r) ? "doubt" : ""} ${isChanged(r) ? "changed" : ""}">
        <td class="mono">${num(top, 2)}</td><td class="mono">${num(bottom, 2)}</td><td class="mono">${num(bottom - top, 2)} м</td>
        <td><span class="facies-chip"><i style="background:${f.color}"></i><b>${f.code}</b></span> <span class="facies-name">${f.name}</span></td>
        <td><span class="conf"><span class="conf-bar"><i style="width:${r.conf * 100}%"></i></span><span class="mono">${pct(r.conf)}</span></span></td>
        <td><select aria-label="Решение геолога">${options(w.edits.get(r.from) ?? r.value)}</select></td>
      </tr>`;
  }).join("") || `<tr class="empty"><td colspan="6">Таких интервалов нет</td></tr>`;

  const changed = w.edits.size;
  $("#download-depth").textContent = changed ? `Скачать результат · правок: ${changed}` : "Скачать результат";
}

function applyEdit(from, value) {
  const w = well();
  const interval = w.intervals.find((r) => r.from === from);
  if (value === interval.value) w.edits.delete(from); else w.edits.set(from, value);
  if (w.edits.size) {
    w.final = Int8Array.from(w.pred);
    for (const r of w.intervals) if (w.edits.has(r.from)) w.final.fill(w.edits.get(r.from), r.from, r.to + 1);
  } else w.final = null;
  redrawPlotKeepingPlace();
  renderComposition();
  renderIntervals();
}

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function downloadDepth() {
  const w = well(), k = w.k, final = w.final ?? w.pred;
  const header = ["Well Name", "Depth", ...FEATURES, ...(w.truth ? ["Facies"] : []), "Predicted_Facies", "Confidence",
    ...FACIES_IDS.map((id) => `P_${id}`), "Geologist_Facies", "Doubtful"];
  const rows = Array.from(w.depth, (d, i) => [w.name, d, ...w.x[i], ...(w.truth ? [w.truth[i]] : []), w.pred[i], w.conf[i],
    ...Array.from(w.proba.subarray(i * k, i * k + k)), final[i], w.conf[i] < state.threshold ? 1 : 0]);
  download(`${w.name}_фации.csv`, toCSV(header, rows, state.source.table.format));
}

function downloadIntervals() {
  const w = well();
  const header = ["Кровля, м", "Подошва, м", "Мощность, м", "Прогноз", "Уверенность", "Сомнительно", "Решение геолога"];
  const rows = w.intervals.map((r) => {
    const top = w.depth[r.from], bottom = w.depth[r.to] + w.step;
    return [top, bottom, bottom - top, faciesLabel(r.value), r.conf, r.conf < state.threshold ? "да" : "нет", faciesLabel(w.edits.get(r.from) ?? r.value)];
  });
  download(`${w.name}_интервалы.csv`, toCSV(header, rows, state.source.table.format));
}

// ---------- сравнение с керном и матрица ошибок ----------
function renderCompare() {
  const w = well();
  $("#compare-panel").hidden = !w.truth;
  if (!w.truth) return;
  const cm = FACIES_IDS.map(() => FACIES_IDS.map(() => 0));
  for (let i = 0; i < w.pred.length; i++) cm[w.truth[i] - 1][w.pred[i] - 1]++;
  renderMatrix($("#compare-matrix"), cm);
}

function renderMatrix(el, cm) {
  const head = `<span class="mx-corner">керн ↓ · прогноз →</span>` + FACIES_IDS.map((k) => `<span class="mx-head" title="${esc(faciesLabel(k))}">${FACIES[k].code}</span>`).join("");
  const body = cm.map((row, r) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return `<span class="mx-head row" title="${esc(faciesLabel(r + 1))}"><i style="background:${FACIES[r + 1].color}"></i>${FACIES[r + 1].code}</span>` +
      row.map((v, c) => `<span class="mx-cell${r === c ? " diag" : ""}${v ? "" : " zero"}" style="--v:${(v / total).toFixed(3)};--d:${r + c}"
        title="Керн ${FACIES[r + 1].code}, прогноз ${FACIES[c + 1].code}: ${v} (${pct(v / total)})">${v || "·"}</span>`).join("");
  }).join("");
  el.innerHTML = head + body;
}

// ---------- раздел «О модели» ----------
function renderModelView(info) {
  const h = info.holdout;
  $("#model-lead").textContent = `Модель обучена на ${info.train_wells.length} скважинах (${num(info.training_rows, 0)} интервалов) и проверена на ${h.holdout_wells.length} скважинах, которых не видела: ${h.holdout_wells.join(", ")} (${h.holdout_rows} интервалов).`;
  const cards = [
    { key: "acc", label: "Точное совпадение с керном", text: pct(h.accuracy) },
    { key: "adj", label: "С точностью до соседней фации", text: pct(h.accuracy_adjacent), tone: "good", sub: "соседние фации геологически близки" },
    { key: "f1", label: "Macro F1", text: h.macro_f1.toFixed(2).replace(".", ","), sub: "редкие фации весят как частые" },
  ];
  $("#model-metrics").innerHTML = cards.map(metricCard).join("");
  renderMatrix($("#model-matrix"), h.confusion_matrix);

  const importance = Object.entries(info.feature_importance).sort((a, b) => b[1] - a[1]);
  const top = importance[0][1];
  $("#model-importance").innerHTML = importance.map(([name, v], i) => `
    <li style="--i:${i}"><span class="bar-label"><b>${name}</b> ${FIELD_HELP[name]}</span>
      <span class="bar-value">${num(v, 1)}</span><span class="bar"><i style="--w:${(v / top) * 100}%"></i></span></li>`).join("");

  $("#model-recall").innerHTML = FACIES_IDS.map((k, i) => {
    const v = h.recall[k];
    return `<li style="--i:${i}"><span class="bar-label"><i style="background:${FACIES[k].color}"></i><b>${FACIES[k].code}</b> ${FACIES[k].name}</span>
      <span class="bar-value">${pct(v)}</span><span class="bar"><i style="--w:${v * 100}%;background:${FACIES[k].color}"></i></span></li>`;
  }).join("");
}

function renderFormatView() {
  const need = { "Well Name": "нет, если скважина одна", Facies: "нет — для сравнения с моделью", Formation: "нет", RELPOS: "да, или столбец с пластом" };
  $("#format-fields").innerHTML = FIELDS.map((f) => `
    <tr><td><code>${f}</code></td><td>${FIELD_HELP[f]}</td><td>${need[f] ?? "да"}</td></tr>`).join("");
}

// ---------- сопоставление столбцов ----------
const dialog = $("#mapping");
let mappingSource = null;
function openMapping(source) {
  mappingSource = source;
  const options = (selected) => [`<option value="">— нет —</option>`, ...source.table.columns.map((c) =>
    `<option${c === selected ? " selected" : ""}>${esc(c)}</option>`)].join("");
  $("#mapping-fields").innerHTML = FIELDS.map((f) => `
    <label class="field"><span><b>${f}</b>${REQUIRED.includes(f) ? "" : "<em>необязательно</em>"}</span>
      <small>${FIELD_HELP[f]}</small>
      <select data-field="${f}">${options(source.mapping[f])}</select></label>`).join("");
  $("#mapping-log").checked = source.logRes;
  updateMappingStatus();
  dialog.showModal();
}
const readMapping = () => Object.fromEntries($$("#mapping-fields select").map((s) => [s.dataset.field, s.value]));
function updateMappingStatus() {
  const missing = missingFields(readMapping());
  $$("#mapping-fields .field").forEach((el) => el.classList.toggle("missing", missing.includes($("select", el).dataset.field)));
  $("#mapping-status").innerHTML = missing.length
    ? `Не удалось найти столбцы: <b>${missing.join(", ")}</b>. Укажите, где они в вашем файле.`
    : "Все нужные столбцы найдены. Проверьте соответствие и нажмите «Применить».";
  $("#mapping-status").classList.toggle("error", missing.length > 0);
  $("#mapping-apply").disabled = missing.length > 0;
}
$("#mapping-fields").addEventListener("change", updateMappingStatus);
dialog.addEventListener("close", () => {
  if (dialog.returnValue !== "apply") return;
  compute({ ...mappingSource, mapping: readMapping(), logRes: $("#mapping-log").checked });
});

// ---------- навигация и события ----------
function showView(name) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
}
$$(".tab").forEach((t) => t.addEventListener("click", () => { showView(t.dataset.view); window.scrollTo({ top: 0 }); }));
$$("[data-goto]").forEach((b) => b.addEventListener("click", () => { showView(b.dataset.goto); window.scrollTo({ top: 0 }); }));

const fileInput = $("#file");
const pickFile = () => fileInput.click();
fileInput.addEventListener("change", () => { if (fileInput.files[0]) openFile(fileInput.files[0]); fileInput.value = ""; });
$("#dropzone").addEventListener("click", pickFile);
$("#dropzone").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickFile(); } });
$("#new-file").addEventListener("click", pickFile);
$("#try-example").addEventListener("click", openExample);
$("#open-mapping").addEventListener("click", () => openMapping(state.source));

// перетаскивание файла в любое место страницы
let dragDepth = 0;
const veil = $("#drag-veil");
addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { dragDepth++; veil.classList.add("show"); } });
addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; veil.classList.remove("show"); } });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0; veil.classList.remove("show");
  if (e.dataTransfer.files[0]) openFile(e.dataTransfer.files[0]);
});

$("#wells").addEventListener("click", (e) => {
  const chip = e.target.closest("[data-i]");
  if (!chip) return;
  state.current = +chip.dataset.i; state.filter = "all";
  renderWells(); renderWell(true);
});

$("#threshold").addEventListener("input", (e) => {
  state.threshold = +e.target.value;
  $("#threshold-out").textContent = pct(state.threshold);
  state.plot.setThreshold(state.threshold);
  $('#metrics [data-key="doubt"] .metric-value').textContent = pct(doubtfulShare(well()));
  renderIntervals();
});

// при прокрутке страницы подсказка у курсора больше не актуальна
addEventListener("scroll", () => { showTip(-1); $(".plot-body.hover")?.classList.remove("hover"); }, { passive: true });

const zoom = (factor) => { state.zoom = Math.min(Math.max(state.zoom * factor, 0.5), 6); redrawPlotKeepingPlace(); };
$("#zoom-in").addEventListener("click", () => zoom(1.5));
$("#zoom-out").addEventListener("click", () => zoom(1 / 1.5));

$("#interval-filter").addEventListener("click", (e) => {
  const chip = e.target.closest("[data-filter]");
  if (chip) { state.filter = chip.dataset.filter; renderIntervals(); }
});
$("#intervals").addEventListener("change", (e) => {
  if (e.target.matches("select")) applyEdit(+e.target.closest("tr").dataset.from, +e.target.value);
});
$("#intervals").addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-from]");
  if (row && !e.target.closest("select")) state.plot.focus(+row.dataset.from, +row.dataset.to);
});
$("#download-depth").addEventListener("click", downloadDepth);
$("#download-intervals").addEventListener("click", downloadIntervals);

function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.textContent = message;
  $("#toasts").append(el);
  setTimeout(() => { el.classList.add("out"); el.addEventListener("animationend", () => el.remove()); }, 6000);
}

// ---------- декоративные «пласты» на первом экране ----------
function buildStrata() {
  const W = 1600, H = 420;
  const layers = [
    { color: FACIES[1].color, base: 70, amp: 18, period: 800, time: 46 },
    { color: FACIES[2].color, base: 130, amp: 22, period: 640, time: 38 },
    { color: FACIES[3].color, base: 195, amp: 16, period: 800, time: 54 },
    { color: FACIES[5].color, base: 255, amp: 24, period: 533.33, time: 42 },
    { color: FACIES[6].color, base: 320, amp: 14, period: 800, time: 60 },
    { color: FACIES[8].color, base: 375, amp: 18, period: 640, time: 50 },
  ];
  const paths = layers.map((l, k) => {
    let d = `M0,${H}`;
    for (let x = 0; x <= W + l.period; x += 20) d += `L${x},${(l.base + l.amp * Math.sin((x / l.period) * Math.PI * 2 + k)).toFixed(1)}`;
    d += `L${W + l.period},${H}Z`;
    return `<path d="${d}" fill="${l.color}" style="--shift:-${l.period}px;--time:${l.time}s"/>`;
  }).join("");
  $("#strata").innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMax slice">${paths}</svg>`;
}

buildStrata();
renderFormatView();
