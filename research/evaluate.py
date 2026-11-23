"""Сравнение по целым скважинам. Запуск из корня: python research/evaluate.py."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from facies import FEATURES, FACIES

OUT = ROOT / "research"
df = pd.read_csv(ROOT / "data/facies_data.csv")
info = json.loads((ROOT / "docs/model/info.json").read_text())
dev = df[df["Well Name"].isin(info["train_wells"])]
classes = list(FACIES)
params = dict(info["parameters"], verbose=False, allow_writing_files=False, thread_count=2)
cat = CatBoostClassifier(**params)
models = {
    "CatBoost": (cat, FEATURES),
    "CatBoost SqrtBalanced": (CatBoostClassifier(**params, auto_class_weights="SqrtBalanced"), FEATURES),
    "ГИС": (cat, FEATURES[:5]),
    "ГИС + RELPOS": (cat, FEATURES[:6]),
    "Логистическая регрессия": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), FEATURES),
    "SVM": (make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf")), FEATURES),
    "kNN": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)), FEATURES),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=2), FEATURES),
    "Extra Trees": (ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=2), FEATURES),
    "XGBoost": (XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.05, random_state=42, n_jobs=2), FEATURES),
}
folds = []
for name in sorted(dev["Well Name"].unique()):
    test = dev[dev["Well Name"] == name]
    train = dev[dev["Well Name"] != name]
    result = test[["Well Name", "Depth", "Facies"]].copy()
    labels, target = np.unique(train.Facies, return_inverse=True)
    votes = []
    for key, (model, cols) in models.items():
        fitted = clone(model).fit(train[cols], target)
        result[key] = labels[fitted.predict(test[cols]).astype(int).ravel()]
        if key in ("CatBoost", "Extra Trees", "XGBoost"):
            votes.append(fitted.predict_proba(test[cols]))
    result["Ансамбль"] = labels[np.mean(votes, axis=0).argmax(axis=1)]
    folds.append(result)
    print(name, flush=True)

oof = pd.concat(folds)
oof.to_csv(OUT / "oof.csv", index=False)
scores, by_class, by_well = [], [], []
for key in [*models, "Ансамбль"]:
    for well, group in oof.groupby("Well Name"):
        by_well.append((key, well, len(group), accuracy_score(group.Facies, group[key])))
    precision, recall, f1, support = precision_recall_fscore_support(oof.Facies, oof[key], labels=classes, zero_division=0)
    by_class.extend((key, c, int(n), p, r, f) for c,n,p,r,f in zip(classes,support,precision,recall,f1))
    scores.append((key, f1.mean(), accuracy_score(oof.Facies, oof[key]), np.mean([a for k,w,n,a in by_well if k==key])))
for file, rows, columns in [
    ("models", scores, ["model","macro_f1","accuracy","mean_well_accuracy"]),
    ("classes", by_class, ["model","facies","points","precision","recall","f1"]),
    ("wells", by_well, ["model","well","points","accuracy"]),
]:
    pd.DataFrame(rows, columns=columns).to_csv(OUT / (file+".csv"), index=False, float_format="%.5f")

# Отдельная описательная проверка двух заранее заданных вариантов CatBoost.
hold = df[df["Well Name"].isin(info["holdout"]["holdout_wells"])]
check = hold[["Well Name", "Depth", "Facies"]].copy()
for key in ["CatBoost", "CatBoost SqrtBalanced"]:
    fitted = clone(models[key][0]).fit(dev[FEATURES], dev.Facies)
    check[key] = fitted.predict(hold[FEATURES]).astype(int).ravel()
check.to_csv(OUT / "holdout.csv", index=False)
