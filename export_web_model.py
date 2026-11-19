"""Выгрузка модели CatBoost для сайта (расчёт идёт в браузере, на JavaScript).

Что делает:
1. Сохраняет деревья модели в компактный JSON  → docs/model/model.json
2. Копирует описание модели и метрики, добавляет
   связь уверенности с точностью               → docs/model/info.json
3. Готовит пример скважин и шаблон             → docs/data/
4. Проверяет, что расчёт «вручную» по этому JSON совпадает с CatBoost,
   и пишет эталон для сверки в браузере        → docs/_ref.json (не попадает в git)

Запуск:  .venv/bin/python export_web_model.py
"""
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from facies import ADJACENT, FEATURES

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "docs"
(WEB / "model").mkdir(parents=True, exist_ok=True)
(WEB / "data").mkdir(parents=True, exist_ok=True)

model = CatBoostClassifier()
model.load_model(str(ROOT / "models" / "catboost_depth4.cbm"))
info = json.loads((ROOT / "models" / "catboost_depth4_info.json").read_text(encoding="utf-8"))

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "model.json"
    model.save_model(str(path), format="json")
    raw = json.loads(path.read_text())

names = [f["feature_id"] for f in raw["features_info"]["float_features"]]
assert names == FEATURES, names
classes = [int(c) for c in model.classes_]
scale, bias = raw["scale_and_bias"]

# Дерево: f — номера признаков, b — пороги, v — значения в листьях
# (лист * число_классов + класс). Бит i номера листа = 1, если признак f[i] > b[i].
trees = [{
    "f": [s["float_feature_index"] for s in t["splits"]],
    "b": [s["border"] for s in t["splits"]],
    # 5 значащих цифр: файл на треть меньше, прогнозы те же (проверка ниже)
    "v": [float(f"{v:.5g}") for v in t["leaf_values"]],
} for t in raw["oblivious_trees"]]

compact = {"features": FEATURES, "classes": classes, "scale": scale, "bias": bias, "trees": trees}
(WEB / "model" / "model.json").write_text(json.dumps(compact, separators=(",", ":")))


def predict_proba(x):
    """Тот же алгоритм, что в docs/js/model.js."""
    x = np.asarray(x, dtype=np.float32)
    k = len(classes)
    out = np.zeros((len(x), k))
    for t in trees:
        leaf = np.zeros(len(x), dtype=int)
        for i, (f, b) in enumerate(zip(t["f"], t["b"])):
            leaf |= (x[:, f] > np.float32(b)).astype(int) << i
        out += np.asarray(t["v"]).reshape(-1, k)[leaf]
    out = out * scale + np.asarray(bias)
    e = np.exp(out - out.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


df = pd.read_csv(ROOT / "data" / "facies_data.csv")
want = model.predict_proba(df[FEATURES])
got = predict_proba(df[FEATURES].to_numpy())
diff = float(np.abs(want - got).max())
same = int((want.argmax(1) == got.argmax(1)).sum())
print(f"Сверка с CatBoost: макс. расхождение вероятностей {diff:.2e}, "
      f"совпало прогнозов {same} из {len(df)}")
assert diff < 1e-3 and same == len(df)

# Насколько уверенности модели можно верить: на отложенных скважинах считаем,
# как часто прогноз верен при разной уверенности. Сайт показывает это в «О модели».
hold = df["Well Name"].isin(info["holdout"]["holdout_wells"]).to_numpy()
conf = want[hold].max(axis=1)
pred = np.asarray(classes)[want[hold].argmax(axis=1)]
truth = df.loc[hold, "Facies"].to_numpy()
exact = pred == truth
near = np.array([p == t or p in ADJACENT[t] for t, p in zip(truth, pred)])
edges = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
info["holdout"]["confidence_bins"] = [{
    "from": lo, "to": hi,
    "n": int(sel.sum()), "exact": int(exact[sel].sum()), "near": int(near[sel].sum()),
} for lo, hi in zip(edges, edges[1:]) for sel in [(conf >= lo) & ((conf < hi) | (hi == 1))]]
(WEB / "model" / "info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

(WEB / "_ref.json").write_text(json.dumps({
    "x": df[FEATURES].to_numpy().tolist(),
    "proba": np.round(want, 7).tolist(),
}))

example = df[df["Well Name"].isin(info["holdout"]["holdout_wells"])]
example.to_csv(WEB / "data" / "example.csv", index=False)
template = pd.read_csv(ROOT / "data" / "new_well.csv").head(40).rename(columns={"Depth": "Depth_m"})
template["Depth_m"] = (template["Depth_m"] * 0.3048).round(6)
template.to_csv(WEB / "data" / "template.csv", index=False)

size = (WEB / "model" / "model.json").stat().st_size / 1024
print(f"Модель для сайта: {size:.0f} КБ, деревьев: {len(trees)}, классы: {classes}")
