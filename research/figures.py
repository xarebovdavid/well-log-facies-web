"""Графики для отчёта. Запускать после evaluate.py."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from facies import FEATURES, FACIES

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
codes = [v[0] for v in FACIES.values()]
classes = pd.read_csv(OUT / "classes.csv")
f1 = classes.pivot(index="model", columns="facies", values="f1")
fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
heat = ax.imshow(f1, aspect="auto", cmap="Blues", vmin=0, vmax=1)
for (i,j), v in np.ndenumerate(f1.values):
    ax.text(j,i,f"{v:.2f}", ha="center", va="center", color="white" if v>.55 else "black")
ax.set_xticks(range(9), codes)
ax.set_yticks(range(len(f1)), f1.index)
ax.set_title("F1 по фациям при переносе на другую скважину", loc="left")
fig.savefig(OUT / "facies.png", dpi=160)
plt.close(fig)

d = pd.read_csv(ROOT / "data/facies_data.csv")
target = d[d["Well Name"]=="CROSS H CATTLE"].copy()
model = CatBoostClassifier()
model.load_model(str(ROOT / "models/catboost_depth4.cbm"))
target["pred"] = model.predict(target[FEATURES]).astype(int).ravel()
names = [n for n,g in d.groupby("Well Name") if {1,2}.issubset(set(g.Facies))]
fig, axes = plt.subplots(1, 2, figsize=(11,4.4), layout="constrained")
colors = ["#246372", "#b27642"]
ax = axes[0]
for i,c in enumerate([1,2]):
    positions = np.arange(len(names)) + (i-.5)*.33
    groups = [d[(d["Well Name"]==w)&(d.Facies==c)].GR for w in names]
    boxes = ax.boxplot(groups, positions=positions, widths=.27, patch_artist=True, showfliers=False,
                       medianprops={"color":"black"})
    for b in boxes["boxes"]: b.set_facecolor(colors[i])
    boxes["boxes"][0].set_label(codes[c-1])
ax.set_xticks(range(len(names)), [n.replace(" ","\n",1) for n in names])
ax.set_ylabel("GR, API")
ax.set_title("Распределение GR по разметке")
ax.legend()
ax=axes[1]
for cls, label, color in [(2,"Керн CSiS",colors[1]),(1,"Керн SS",colors[0])]:
    sub=target[target.Facies==cls]
    ax.scatter(sub.GR,sub.PHIND,s=12,alpha=.45,label=label,color=color)
err=target[(target.Facies==1)&(target.pred==2)]
ax.scatter(err.GR,err.PHIND,s=27,facecolors="none",edgecolors="#262624",linewidth=.6,label=f"Ошибка SS → CSiS (n={len(err)})")
ax.set(xlabel="GR, API",ylabel="PHIND, %",title="CROSS H CATTLE")
ax.legend(fontsize=8)
fig.savefig(OUT / "sandstone.png", dpi=160)
plt.close(fig)

counts = pd.crosstab(d["Well Name"],d.Facies)
fig, ax = plt.subplots(figsize=(9,4),layout="constrained")
left=np.zeros(len(counts))
for c in counts.columns:
    pct=100*counts[c]/counts.sum(axis=1)
    ax.barh(counts.index,pct,left=left,label=codes[c-1],color=FACIES[c][2])
    left+=pct.to_numpy()
ax.set(xlabel="Доля размеченных точек, %",xlim=(0,100),title="Состав выборки по скважинам")
ax.legend(ncol=5,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.18))
fig.savefig(OUT / "wells.png", dpi=160)
plt.close(fig)
