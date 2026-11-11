// Расчёт модели CatBoost в браузере.
// Формат model.json и алгоритм описаны в export_web_model.py — там же сверка с Python.

export async function loadModel(url) {
  const raw = await (await fetch(url)).json();
  return {
    features: raw.features,
    classes: raw.classes,
    scale: raw.scale,
    bias: raw.bias,
    trees: raw.trees.map((t) => ({
      f: Int8Array.from(t.f),
      b: Float32Array.from(t.b), // CatBoost сравнивает числа одинарной точности
      v: Float64Array.from(t.v),
    })),
  };
}

/** rows — массив строк по 7 признаков. Возвращает прогноз, уверенность и вероятности (n × k подряд). */
export function predict(model, rows) {
  const n = rows.length;
  const k = model.classes.length;
  const proba = new Float64Array(n * k);
  const pred = new Int8Array(n);
  const conf = new Float64Array(n);
  const x = new Float32Array(model.features.length);
  const sum = new Float64Array(k);

  for (let r = 0; r < n; r++) {
    x.set(rows[r]);
    sum.fill(0);
    for (const t of model.trees) {
      let leaf = 0;
      for (let i = 0; i < t.f.length; i++) if (x[t.f[i]] > t.b[i]) leaf |= 1 << i;
      const at = leaf * k;
      for (let c = 0; c < k; c++) sum[c] += t.v[at + c];
    }
    // softmax
    let top = -Infinity;
    for (let c = 0; c < k; c++) {
      sum[c] = sum[c] * model.scale + model.bias[c];
      if (sum[c] > top) top = sum[c];
    }
    let total = 0;
    for (let c = 0; c < k; c++) total += sum[c] = Math.exp(sum[c] - top);
    let best = 0;
    for (let c = 0; c < k; c++) {
      const p = sum[c] / total;
      proba[r * k + c] = p;
      if (p > proba[r * k + best]) best = c;
    }
    pred[r] = model.classes[best];
    conf[r] = proba[r * k + best];
  }
  return { pred, conf, proba, k };
}
