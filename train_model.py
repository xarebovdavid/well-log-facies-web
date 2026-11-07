"""Обучение модели для сайта.

Повторяет train.py из репозитория команды (те же признаки, параметры
CatBoost и разбиение по скважинам), а дополнительно сохраняет метрики
на отложенных скважинах — их показывает вкладка «О модели».

Запуск:  .venv/bin/python train_model.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from facies import ADJACENT, FEATURES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "facies_data.csv"
MODEL_PATH = ROOT / "models" / "catboost_depth4.cbm"
INFO_PATH = ROOT / "models" / "catboost_depth4_info.json"

PARAMS = dict(iterations=500, depth=4, learning_rate=0.05, loss_function="MultiClass", random_seed=42)

df = pd.read_csv(DATA_PATH)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
dev_idx, hold_idx = next(splitter.split(df, df["Facies"], df["Well Name"]))
dev, hold = df.iloc[dev_idx], df.iloc[hold_idx]

# 1) Честная проверка: учимся только на скважинах разработки,
#    проверяем на скважинах, которых модель не видела.
check = CatBoostClassifier(**PARAMS, verbose=False, allow_writing_files=False, thread_count=2)
check.fit(dev[FEATURES], dev["Facies"])
y_true = hold["Facies"].to_numpy()
y_pred = np.asarray(check.predict(hold[FEATURES])).ravel().astype(int)
adjacent_ok = np.array([p == t or p in ADJACENT[t] for t, p in zip(y_true, y_pred)])
classes = list(range(1, 10))

metrics = {
    "holdout_wells": sorted(hold["Well Name"].unique().tolist()),
    "holdout_rows": int(len(hold)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "accuracy_adjacent": float(adjacent_ok.mean()),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
    "recall": {str(c): float(r) for c, r in zip(classes, recall_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0))},
    "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
}

# 2) Модель для сайта — та же, что в train.py команды (скважины разработки).
model = check
model.save_model(str(MODEL_PATH))

info = {
    "algorithm": "CatBoostClassifier",
    "features": FEATURES,
    "train_wells": sorted(dev["Well Name"].unique().tolist()),
    "training_rows": int(len(dev)),
    "parameters": PARAMS,
    "feature_importance": dict(zip(FEATURES, map(float, model.get_feature_importance()))),
    "holdout": metrics,
}
INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

print("Скважины обучения:", info["train_wells"])
print("Отложенные скважины:", metrics["holdout_wells"])
print(f"Точность: {metrics['accuracy']:.3f}  с учётом соседних фаций: {metrics['accuracy_adjacent']:.3f}"
      f"  macro F1: {metrics['macro_f1']:.3f}")
print("Модель сохранена:", MODEL_PATH)
