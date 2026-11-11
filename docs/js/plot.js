// Планшет скважины: кривые ГИС, уверенность модели и колонки фаций.
// Рисуется в SVG: по горизонтали 0–100 (растягивается на ширину трека), по вертикали — пиксели.
import { FACIES, FACIES_IDS, FEATURES } from "./facies.js";

const col = (name) => FEATURES.indexOf(name);
const TRACKS = [
  { title: "GR", unit: "API", series: [{ name: "GR", col: col("GR"), color: "var(--c-gr)" }] },
  { title: "ILD", unit: "log₁₀ Ом·м", series: [{ name: "ILD", col: col("ILD_log10"), color: "var(--c-ild)" }] },
  { title: "Пористость", unit: "%", series: [
    { name: "PHIND", col: col("PHIND"), color: "var(--c-phi)" },
    { name: "ΔPHI", col: col("DeltaPHI"), color: "var(--c-dphi)", dashed: true },
  ] },
  { title: "PE", unit: "", series: [{ name: "PE", col: col("PE"), color: "var(--c-pe)" }] },
  { title: "Уверенность", unit: "", range: [0, 1], series: [{ name: "Уверенность", conf: true, color: "var(--c-conf)" }] },
];

const fmt = (v, digits = 2) => v.toLocaleString("ru-RU", { maximumFractionDigits: digits, useGrouping: false });

/** Непрерывные участки одинаковых значений: [{ from, to, value }] (to — включительно). */
export function runs(values, depth, step) {
  const out = [];
  let from = 0;
  for (let i = 1; i <= values.length; i++) {
    const gap = i < values.length && depth[i] - depth[i - 1] > step * 1.5;
    if (i === values.length || values[i] !== values[from] || gap) {
      out.push({ from, to: i - 1, value: values[from] });
      from = i;
    }
  }
  return out;
}

export function createPlot(root, well, { threshold, pxPerMeter, animate, onHover }) {
  const { depth, x, conf, pred, truth, final, step } = well;
  const n = depth.length;
  const d0 = depth[0] - step / 2, d1 = depth[n - 1] + step / 2;
  const H = Math.round((d1 - d0) * pxPerMeter);
  const y = (d) => (d - d0) * pxPerMeter;

  const strips = [{ title: "Прогноз", values: pred }];
  if (truth) strips.push({ title: "Керн", values: truth });
  if (final) strips.push({ title: "Итог", values: final });

  // --- шкалы кривых ---
  const tracks = TRACKS.map((t) => {
    const get = t.series.map((s) => (s.conf ? (i) => conf[i] : (i) => x[i][s.col]));
    let lo = Infinity, hi = -Infinity;
    if (t.range) [lo, hi] = t.range;
    else {
      for (const g of get) for (let i = 0; i < n; i++) { const v = g(i); if (v < lo) lo = v; if (v > hi) hi = v; }
      const pad = (hi - lo || 1) * 0.04;
      lo -= pad; hi += pad;
    }
    return { ...t, get, lo, hi, px: (v) => ((v - lo) / (hi - lo)) * 100 };
  });

  // --- деления по глубине ---
  const tickStep = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500].find((s) => s * pxPerMeter >= 64) ?? 1000;
  const ticks = [];
  for (let d = Math.ceil(d0 / tickStep) * tickStep; d <= d1; d += tickStep) ticks.push(d);

  const grid = ticks.map((d) => `<line x1="0" x2="100" y1="${y(d).toFixed(1)}" y2="${y(d).toFixed(1)}"/>`).join("") +
    [25, 50, 75].map((v) => `<line x1="${v}" x2="${v}" y1="0" y2="${H}"/>`).join("");

  const curvePaths = (t) => t.series.map((s, k) => {
    let line = "", area = "", segStart = -1;
    const closeArea = (i) => { area += `L0,${y(depth[i]).toFixed(1)}L0,${y(depth[segStart]).toFixed(1)}Z`; };
    for (let i = 0; i < n; i++) {
      const p = `${t.px(t.get[k](i)).toFixed(2)},${y(depth[i]).toFixed(1)}`;
      if (i === 0 || depth[i] - depth[i - 1] > step * 1.5) {
        if (segStart >= 0) closeArea(i - 1);
        segStart = i;
        line += "M" + p; area += "M" + p;
      } else { line += "L" + p; area += "L" + p; }
    }
    closeArea(n - 1);
    return (s.dashed ? "" : `<path class="area" d="${area}" style="fill:${s.color}"/>`) +
      `<path class="curve${s.dashed ? " dashed" : ""}" d="${line}" style="stroke:${s.color}"/>`;
  }).join("");

  const stripRects = (values) => runs(values, depth, step).map((r) =>
    `<rect x="0" width="100" y="${y(depth[r.from] - step / 2).toFixed(1)}" height="${((depth[r.to] - depth[r.from] + step) * pxPerMeter + 0.4).toFixed(1)}" fill="${FACIES[r.value].color}"/>`).join("");

  const svg = (inner) => `<svg viewBox="0 0 100 ${H}" preserveAspectRatio="none" aria-hidden="true">${inner}</svg>`;
  const template = `56px repeat(${tracks.length}, minmax(0, 1fr)) repeat(${strips.length}, 46px)`;

  root.innerHTML = `
    <div class="plot-head" style="grid-template-columns:${template}">
      <div class="th th-depth">Глубина<small>м</small></div>
      ${tracks.map((t) => `
        <div class="th">
          <div class="th-title">${t.series.map((s) => `<i style="background:${s.color}"></i>`).join("")}${t.title}${t.unit ? `<small>${t.unit}</small>` : ""}</div>
          <div class="th-scale"><span>${fmt(t.lo, 1)}</span><span>${fmt(t.hi, 1)}</span></div>
        </div>`).join("")}
      ${strips.map((s) => `<div class="th th-strip">${s.title}</div>`).join("")}
    </div>
    <div class="plot-body${animate ? " animate" : ""}" style="grid-template-columns:${template};height:${H}px">
      <div class="axis">${ticks.map((d) => `<span style="top:${y(d).toFixed(1)}px">${d}</span>`).join("")}</div>
      ${tracks.map((t, i) => `
        <div class="track" style="--i:${i};grid-column:${i + 2}">
          ${svg(`<g class="grid">${grid}</g>${curvePaths(t)}`)}
          ${t.series.map((s) => `<b class="dot" style="background:${s.color}"></b>`).join("")}
          ${t.range ? `<div class="threshold-line"></div>` : ""}
        </div>`).join("")}
      ${strips.map((s, i) => `<div class="strip" style="--i:${tracks.length + i};grid-column:${tracks.length + i + 2}">${svg(stripRects(s.values))}</div>`).join("")}
      <div class="bands" style="grid-column:2 / ${2 + tracks.length}"></div>
      <div class="cursor"></div>
    </div>`;

  const body = root.querySelector(".plot-body");
  if (animate) setTimeout(() => body.classList.remove("animate"), 2400);
  const bands = root.querySelector(".bands");
  const cursor = root.querySelector(".cursor");
  const thresholdLine = root.querySelector(".threshold-line");
  const trackEls = [...root.querySelectorAll(".track")];

  function setThreshold(t) {
    const low = Array.from(conf, (c) => (c < t ? 1 : 0));
    bands.innerHTML = runs(low, depth, step).filter((r) => r.value).map((r) =>
      `<i style="top:${y(depth[r.from] - step / 2).toFixed(1)}px;height:${((depth[r.to] - depth[r.from] + step) * pxPerMeter).toFixed(1)}px"></i>`).join("");
    thresholdLine.style.left = t * 100 + "%";
  }
  setThreshold(threshold);

  // --- курсор: ближайшая к указателю глубина ---
  let frame = 0, last = -1;
  function nearest(d) {
    let lo = 0, hi = n - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (depth[mid] < d) lo = mid + 1; else hi = mid; }
    return lo > 0 && d - depth[lo - 1] < depth[lo] - d ? lo - 1 : lo;
  }
  function move(e) {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const top = e.clientY - body.getBoundingClientRect().top;
      const i = nearest(d0 + top / pxPerMeter);
      const py = y(depth[i]);
      body.classList.add("hover");
      cursor.style.transform = `translateY(${py}px)`;
      trackEls.forEach((el, k) => el.querySelectorAll(".dot").forEach((dot, s) => {
        dot.style.left = tracks[k].px(tracks[k].get[s](i)) + "%";
        dot.style.top = py + "px";
      }));
      if (i !== last || e.type === "pointermove") onHover?.(i, e);
      last = i;
    });
  }
  function leave() {
    cancelAnimationFrame(frame);
    body.classList.remove("hover");
    last = -1;
    onHover?.(-1);
  }
  body.addEventListener("pointermove", move);
  body.addEventListener("pointerleave", leave);

  /** Прокрутить к интервалу и подсветить его. */
  function focus(from, to) {
    const top = y(depth[from] - step / 2), height = (depth[to] - depth[from] + step) * pxPerMeter;
    const flash = document.createElement("div");
    flash.className = "flash";
    flash.style.cssText = `top:${top}px;height:${Math.max(height, 3)}px`;
    body.append(flash);
    flash.addEventListener("animationend", () => flash.remove());
    const target = body.getBoundingClientRect().top + window.scrollY + top + height / 2 - window.innerHeight / 2;
    window.scrollTo({ top: target, behavior: "smooth" });
  }

  return { setThreshold, focus, height: H };
}

/** Содержимое всплывающей подсказки для глубины i. */
export function tooltipHTML(well, i, threshold) {
  const { depth, x, conf, pred, truth, proba, k } = well;
  const f = pred[i];
  const bars = FACIES_IDS.map((id, c) => {
    const p = proba[i * k + c];
    return `<i title="${FACIES[id].code}" style="height:${Math.max(p * 100, 2).toFixed(0)}%;background:${FACIES[id].color};opacity:${id === f ? 1 : 0.55}"></i>`;
  }).join("");
  const val = (name, digits = 2) => fmt(x[i][col(name)], digits);
  return `
    <div class="tip-depth">${fmt(depth[i], 2)} м</div>
    <div class="tip-facies"><i style="background:${FACIES[f].color}"></i><b>${FACIES[f].code}</b> ${FACIES[f].name}</div>
    <div class="tip-conf ${conf[i] < threshold ? "low" : ""}">уверенность ${Math.round(conf[i] * 100)}%${conf[i] < threshold ? " · сомнительно" : ""}</div>
    <div class="tip-bars">${bars}</div>
    ${truth ? `<div class="tip-core ${truth[i] === f ? "ok" : "miss"}">керн: <b>${FACIES[truth[i]].code}</b> ${truth[i] === f ? "— совпало" : "— расхождение"}</div>` : ""}
    <dl class="tip-values">
      <dt>GR</dt><dd>${val("GR", 1)}</dd><dt>ILD</dt><dd>${val("ILD_log10")}</dd>
      <dt>PHIND</dt><dd>${val("PHIND", 1)}</dd><dt>ΔPHI</dt><dd>${val("DeltaPHI", 1)}</dd>
      <dt>PE</dt><dd>${val("PE")}</dd><dt>NM_M</dt><dd>${val("NM_M", 0)}</dd>
    </dl>`;
}
