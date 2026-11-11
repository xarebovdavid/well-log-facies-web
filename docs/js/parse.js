// Чтение файлов скважин (CSV, Excel, LAS) и подготовка данных для модели.
import { FEATURES } from "./facies.js";

export const NONE = "";
export const REQUIRED = ["Depth", ...FEATURES];
export const FIELDS = ["Well Name", "Depth", ...FEATURES, "Facies", "Formation"];

// Как столбцы могут называться в файлах пользователей.
const ALIASES = {
  "Well Name": ["well name", "well", "wellname", "well_name", "скважина", "скв"],
  Depth: ["depth", "dept", "md", "глубина", "глуб"],
  GR: ["gr", "гк", "gk", "gamma"],
  ILD_log10: ["ild_log10", "ild_log", "lld_log10", "log_ild"],
  DeltaPHI: ["deltaphi", "delta_phi", "dphi"],
  PHIND: ["phind", "phi_nd", "phi"],
  PE: ["pe", "pef"],
  RELPOS: ["relpos", "rel_pos"],
  NM_M: ["nm_m", "nmm", "nm-m"],
  Facies: ["facies", "фация", "фации", "core_facies", "литология", "керн"],
  Formation: ["formation", "пласт", "свита", "fm"],
};
const RAW_RESISTIVITY = ["ild", "lld", "rt", "res", "resd"]; // Ом·м без логарифма

const XLSX_URL = "https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js";

/** Файл → таблица { columns, rows, wellName?, format }. */
export async function readFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  const buffer = await file.arrayBuffer();
  if (ext === "xlsx" || ext === "xls") return parseXLSX(buffer);
  const text = decodeText(buffer);
  if (ext === "las") return parseLAS(text);
  return parseCSV(text);
}

// Русские файлы часто в Windows-1251 — пробуем UTF-8, иначе её.
function decodeText(buffer) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch {
    return new TextDecoder("windows-1251").decode(buffer);
  }
}

export function parseCSV(text) {
  text = text.replace(/^﻿/, "");
  const firstLine = text.slice(0, text.search(/\r?\n/) >>> 0);
  const count = (ch) => firstLine.split(ch).length - 1;
  let delimiter = [";", "\t", ","].sort((a, b) => count(b) - count(a))[0];
  if (count(delimiter) === 0) delimiter = /\s+/; // столбцы через пробелы

  const all = typeof delimiter === "string" ? splitDelimited(text, delimiter)
    : text.split(/\r?\n/).map((l) => l.trim().split(delimiter));
  const rows = all.filter((r) => r.some((c) => c !== ""));
  if (rows.length < 2) throw new Error("В файле нет строк с данными.");
  const columns = rows.shift().map((c) => c.trim());

  // Десятичная запятая: «12,5» при разделителе, отличном от запятой
  const decimalComma = delimiter !== "," &&
    rows.slice(0, 50).some((r) => r.some((c) => /^-?\d+,\d+$/.test(c.trim())));
  return { columns, rows, format: { delimiter: typeof delimiter === "string" ? delimiter : ",", decimalComma } };
}

function splitDelimited(text, d) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === d) { row.push(cell); cell = ""; }
    else if (ch === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (ch !== "\r") cell += ch;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

export function parseLAS(text) {
  let section = "", wellName = "", nullValue = -999.25;
  const columns = [], tokens = [];
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s || s.startsWith("#")) continue;
    if (s.startsWith("~")) { section = s[1].toUpperCase(); continue; }
    if (section === "A") { tokens.push(...s.split(/\s+/)); continue; }
    const m = s.match(/^([^.\s]+)\s*\.(\S*)\s*(.*?)\s*:\s*(.*)$/); // МНЕМОНИКА.ЕД  значение : описание
    if (!m) continue;
    if (section === "C") columns.push(m[1]);
    if (section === "W") {
      const key = m[1].toUpperCase();
      if (key === "WELL") wellName = m[3] || m[4];
      if (key === "NULL" && isFinite(parseFloat(m[3]))) nullValue = parseFloat(m[3]);
    }
  }
  if (!columns.length || !tokens.length) throw new Error("В LAS-файле не найдены секции ~Curve и ~ASCII.");
  const rows = [];
  for (let i = 0; i + columns.length <= tokens.length; i += columns.length) {
    rows.push(tokens.slice(i, i + columns.length).map((t) => (parseFloat(t) === nullValue ? "" : t)));
  }
  return { columns, rows, wellName: wellName.trim(), format: { delimiter: ",", decimalComma: false } };
}

async function parseXLSX(buffer) {
  if (!window.XLSX) {
    await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = XLSX_URL;
      s.onload = resolve;
      s.onerror = () => reject(new Error("Не удалось загрузить модуль чтения Excel (нужен интернет). Сохраните файл как CSV."));
      document.head.append(s);
    });
  }
  const book = window.XLSX.read(buffer);
  const all = window.XLSX.utils.sheet_to_json(book.Sheets[book.SheetNames[0]], { header: 1, defval: "" });
  const rows = all.filter((r) => r.some((c) => c !== ""));
  if (rows.length < 2) throw new Error("На первом листе нет строк с данными.");
  const columns = rows.shift().map((c) => String(c).trim());
  return { columns, rows, format: { delimiter: ";", decimalComma: true } };
}

/** Угадать, какой столбец файла чему соответствует. */
export function guessMapping(columns) {
  const low = new Map(columns.map((c) => [c.trim().toLowerCase(), c]));
  const find = (names) => { for (const n of names) if (low.has(n)) return low.get(n); return NONE; };
  const mapping = {};
  for (const field of FIELDS) mapping[field] = find(ALIASES[field]);
  let logRes = false;
  if (!mapping.ILD_log10) {
    mapping.ILD_log10 = find(RAW_RESISTIVITY);
    logRes = Boolean(mapping.ILD_log10);
  }
  return { mapping, logRes };
}

/** Каких обязательных столбцов не хватает (RELPOS можно посчитать по пластам). */
export function missingFields(mapping) {
  return REQUIRED.filter((f) => !mapping[f] && !(f === "RELPOS" && mapping.Formation));
}

/** Таблица + соответствие столбцов → скважины, готовые к расчёту. */
export function buildDataset(table, mapping, { logRes, fallbackWell }) {
  const at = Object.fromEntries(FIELDS.map((f) => [f, mapping[f] ? table.columns.indexOf(mapping[f]) : -1]));
  const num = (v) => {
    if (typeof v === "number") return v;
    const s = String(v ?? "").trim().replace(",", ".");
    return s === "" ? NaN : Number(s);
  };
  const notes = [];

  let records = table.rows.map((r) => ({
    well: at["Well Name"] >= 0 ? String(r[at["Well Name"]]).trim() : (table.wellName || fallbackWell),
    depth: num(r[at.Depth]),
    formation: at.Formation >= 0 ? String(r[at.Formation]).trim() : "",
    facies: at.Facies >= 0 ? num(r[at.Facies]) : NaN,
    x: FEATURES.map((f) => (at[f] >= 0 ? num(r[at[f]]) : NaN)),
  }));

  const ILD = FEATURES.indexOf("ILD_log10"), RELPOS = FEATURES.indexOf("RELPOS");
  if (logRes) for (const r of records) r.x[ILD] = r.x[ILD] > 0 ? Math.log10(r.x[ILD]) : NaN;

  if (at.RELPOS < 0 && at.Formation >= 0) {
    const span = new Map();
    for (const r of records) {
      if (!isFinite(r.depth)) continue;
      const key = r.well + "\u0000" + r.formation;
      const s = span.get(key) ?? [Infinity, -Infinity];
      span.set(key, [Math.min(s[0], r.depth), Math.max(s[1], r.depth)]);
    }
    for (const r of records) {
      const [top, bottom] = span.get(r.well + "\u0000" + r.formation) ?? [0, 0];
      r.x[RELPOS] = bottom > top ? Math.round((1 - (r.depth - top) / (bottom - top)) * 1000) / 1000 : 1;
    }
    notes.push("RELPOS посчитан по кровле и подошве пластов.");
  }

  const total = records.length;
  records = records.filter((r) => isFinite(r.depth) && r.x.every(isFinite));
  if (records.length < total) notes.push(`Пропущено строк с пустыми или нечисловыми значениями: ${total - records.length}.`);
  if (!records.length) throw new Error("Не осталось ни одной строки с полными данными. Проверьте соответствие столбцов.");

  let hasTruth = at.Facies >= 0;
  if (hasTruth && !records.every((r) => Number.isInteger(r.facies) && r.facies >= 1 && r.facies <= 9)) {
    hasTruth = false;
    notes.push("Столбец фаций по керну содержит значения вне 1–9 — сравнение с керном отключено.");
  }

  const byWell = new Map();
  for (const r of records) (byWell.get(r.well) ?? byWell.set(r.well, []).get(r.well)).push(r);
  const wells = [...byWell].sort(([a], [b]) => a.localeCompare(b, "ru")).map(([name, rs]) => {
    rs.sort((a, b) => a.depth - b.depth);
    const diffs = rs.slice(1).map((r, i) => r.depth - rs[i].depth).filter((d) => d > 0).sort((a, b) => a - b);
    return {
      name,
      depth: Float64Array.from(rs, (r) => r.depth),
      x: rs.map((r) => r.x),
      truth: hasTruth ? Int8Array.from(rs, (r) => r.facies) : null,
      step: diffs.length ? diffs[diffs.length >> 1] : 0.5,
    };
  });
  return { wells, notes, hasTruth };
}

/** Таблица → текст CSV в том же формате, что и исходный файл. */
export function toCSV(header, rows, format) {
  const d = format?.delimiter ?? ",";
  const cell = (v) => {
    if (typeof v === "number") {
      const s = String(Math.round(v * 1e6) / 1e6);
      return format?.decimalComma ? s.replace(".", ",") : s;
    }
    const s = String(v ?? "");
    return s.includes(d) || /["\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return "﻿" + [header, ...rows].map((r) => r.map(cell).join(d)).join("\r\n");
}
