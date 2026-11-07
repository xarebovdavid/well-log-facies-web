"""Сайт: классификация литологии по данным ГИС.

Запуск:  .venv/bin/streamlit run app.py
"""
from io import BytesIO, StringIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier
from plotly.subplots import make_subplots

from facies import ADJACENT, FACIES, FEATURE_HELP, FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "catboost_depth4.cbm"
INFO_PATH = ROOT / "models" / "catboost_depth4_info.json"
EXAMPLE_PATH = ROOT / "data" / "facies_data.csv"

st.set_page_config(page_title="Литология по ГИС", page_icon="🪨", layout="wide")

LABELS = {k: f"{k} · {code} — {name}" for k, (code, name, _) in FACIES.items()}
COLORS = {k: c for k, (_, _, c) in FACIES.items()}

# Как столбцы могут называться в файлах пользователей.
ALIASES = {
    "Well Name": ["well name", "well", "wellname", "well_name", "скважина", "скв"],
    "Depth": ["depth", "dept", "md", "глубина"],
    "GR": ["gr", "гк", "gamma"],
    "ILD_log10": ["ild_log10", "ild_log", "lld_log10", "log_ild"],
    "DeltaPHI": ["deltaphi", "delta_phi", "dphi"],
    "PHIND": ["phind", "phi_nd", "phi"],
    "PE": ["pe", "pef"],
    "RELPOS": ["relpos", "rel_pos"],
    "NM_M": ["nm_m", "nmm", "nm-m"],
    "Facies": ["facies", "фация", "фации", "core_facies", "литология", "керн"],
    "Formation": ["formation", "пласт", "свита", "fm"],
}
RAW_RESISTIVITY = ["ild", "lld", "rt", "res", "resd"]  # Ом·м без логарифма
REQUIRED = ["Depth"] + FEATURES
NONE = "— нет —"


# ---------- загрузка ----------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    return model, info


def read_file(name: str, data: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(BytesIO(data))
    if suffix == ".las":
        import lasio
        las = lasio.read(StringIO(data.decode("utf-8", errors="replace")))
        df = las.df().reset_index()
        well = str(las.well.WELL.value or "").strip() if "WELL" in las.well else ""
        df["Well Name"] = well or Path(name).stem
        return df
    text = data.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(StringIO(text), sep=None, engine="python")
    # Российский формат: «;» и десятичная запятая
    if df.select_dtypes("number").shape[1] < 2 and ";" in text[:500]:
        df = pd.read_csv(StringIO(text), sep=";", decimal=",")
    return df


def guess_mapping(columns):
    low = {c.strip().lower(): c for c in columns}
    mapping, raw_res = {}, False
    for target, names in ALIASES.items():
        mapping[target] = next((low[n] for n in names if n in low), NONE)
    if mapping["ILD_log10"] == NONE:
        hit = next((low[n] for n in RAW_RESISTIVITY if n in low), None)
        if hit:
            mapping["ILD_log10"], raw_res = hit, True
    return mapping, raw_res


def add_relpos(df):
    """RELPOS по кровле/подошве пласта: 1 — кровля, 0 — подошва."""
    g = df.groupby(["Well Name", "Formation"])["Depth"]
    top, bottom = g.transform("min"), g.transform("max")
    span = (bottom - top).replace(0, np.nan)
    return (1 - (df["Depth"] - top) / span).fillna(1.0).round(3)


def predict(df, model):
    proba = model.predict_proba(df[FEATURES])
    classes = [int(c) for c in model.classes_]
    out = df.copy()
    out["Predicted_Facies"] = np.array(classes)[proba.argmax(axis=1)]
    out["Confidence"] = proba.max(axis=1)
    for i, c in enumerate(classes):
        out[f"P_{c}"] = proba[:, i]
    return out


def intervals(well_df, threshold):
    w = well_df.sort_values("Depth").reset_index(drop=True)
    step = float(w["Depth"].diff().median() or 0.5)
    gid = (w["Predicted_Facies"] != w["Predicted_Facies"].shift()).cumsum()
    rows = []
    for _, g in w.groupby(gid):
        f = int(g["Predicted_Facies"].iloc[0])
        conf = float(g["Confidence"].mean())
        rows.append({
            "Кровля, м": float(g["Depth"].min()),
            "Подошва, м": float(g["Depth"].max() + step),
            "Мощность, м": round(len(g) * step, 2),
            "Прогноз": LABELS[f],
            "Уверенность": round(conf, 2),
            "Сомнительно": conf < threshold,
            "Решение геолога": LABELS[f],
        })
    return pd.DataFrame(rows)


# ---------- графика ----------
def facies_strip(depth, facies, name, hover_extra=None):
    scale = []
    for k in range(1, 10):
        lo, hi = (k - 1) / 9, k / 9
        scale += [[lo, COLORS[k]], [hi, COLORS[k]]]
    text = [[LABELS[int(f)] + (hover_extra[i] if hover_extra is not None else "")] for i, f in enumerate(facies)]
    return go.Heatmap(
        z=[[f] for f in facies], y=depth, x=[name], text=text,
        hovertemplate="%{y} м<br>%{text}<extra></extra>",
        colorscale=scale, zmin=0.5, zmax=9.5, showscale=False,
    )


def log_plot(w, threshold, has_truth):
    tracks = [
        ("GR", "GR, API", "#2E7D32"),
        ("ILD_log10", "ILD, log₁₀ Ом·м", "#C62828"),
        ("PHIND", "Пористость, %", "#1565C0"),
        ("PE", "PE", "#6A1B9A"),
        ("Confidence", "Уверенность", "#455A64"),
    ]
    strips = ["Прогноз"] + (["Керн"] if has_truth else [])
    n = len(tracks) + len(strips)
    widths = [1] * len(tracks) + [0.35] * len(strips)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.012,
                        column_widths=widths, subplot_titles=[t[1] for t in tracks] + strips)
    d = w["Depth"]
    for i, (col, _, color) in enumerate(tracks, start=1):
        fig.add_trace(go.Scatter(x=w[col], y=d, mode="lines", line=dict(color=color, width=1.2),
                                 name=col, hovertemplate="%{y} м: %{x:.3f}<extra>" + col + "</extra>"), 1, i)
    # DeltaPHI на одном треке с PHIND
    fig.add_trace(go.Scatter(x=w["DeltaPHI"], y=d, mode="lines", line=dict(color="#90CAF9", width=1, dash="dot"),
                             name="DeltaPHI", hovertemplate="%{y} м: %{x:.2f}<extra>DeltaPHI</extra>"), 1, 3)
    # Подсветка сомнительных интервалов на всех кривых
    low = w["Confidence"] < threshold
    for top, bot in _runs(d.to_numpy(), low.to_numpy()):
        for c in range(1, len(tracks) + 1):
            fig.add_hrect(y0=top, y1=bot, fillcolor="rgba(229,57,53,0.13)", line_width=0, row=1, col=c)
    fig.add_vline(x=threshold, line=dict(color="#E53935", dash="dash", width=1), row=1, col=len(tracks))
    fig.update_xaxes(range=[0, 1], row=1, col=len(tracks))

    extra = [f"<br>уверенность {c:.0%}" for c in w["Confidence"]]
    fig.add_trace(facies_strip(d, w["Predicted_Facies"], "Прогноз", extra), 1, len(tracks) + 1)
    if has_truth:
        fig.add_trace(facies_strip(d, w["Facies"], "Керн"), 1, len(tracks) + 2)
    for c in range(len(tracks) + 1, n + 1):
        fig.update_xaxes(showticklabels=False, row=1, col=c)

    fig.update_yaxes(autorange="reversed", title_text="Глубина, м", row=1, col=1)
    fig.update_xaxes(side="top", showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_layout(height=max(650, min(1600, len(w) * 2)), showlegend=False,
                      margin=dict(l=10, r=10, t=80, b=10), template="plotly_white", hovermode="y")
    fig.update_annotations(font_size=12, yshift=28)
    return fig


def _runs(depth, mask):
    """Непрерывные участки, где mask == True → [(кровля, подошва)]."""
    out, start = [], None
    step = np.median(np.diff(depth)) if len(depth) > 1 else 0.5
    for i, m in enumerate(mask):
        if m and start is None:
            start = depth[i]
        if (not m or i == len(mask) - 1) and start is not None:
            end = depth[i] + step / 2 if m else depth[i - 1] + step / 2
            out.append((start - step / 2, end))
            start = None
    return out


def legend_html():
    chips = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<span style="width:18px;height:14px;border-radius:3px;background:{c};border:1px solid #0002"></span>'
        f'<span style="font-size:0.85rem">{LABELS[k]}</span></div>'
        for k, c in COLORS.items())
    return chips


def confusion_fig(cm, title):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [FACIES[k][0] for k in range(1, 10)]
    fig = go.Figure(go.Heatmap(
        z=norm, x=names, y=names, colorscale="Blues", zmin=0, zmax=1,
        text=cm, texttemplate="%{text}",
        hovertemplate="Керн: %{y}<br>Прогноз: %{x}<br>%{z:.0%} строк класса<extra></extra>"))
    fig.update_layout(title=title, height=460, template="plotly_white",
                      xaxis_title="Прогноз модели", yaxis_title="Фация по керну",
                      yaxis_autorange="reversed", margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------- интерфейс ----------
model, info = load_model()

with st.sidebar:
    st.header("1. Данные")
    upload = st.file_uploader("Файл скважины: CSV, Excel или LAS", type=["csv", "txt", "xlsx", "xls", "las"])
    if st.button("Попробовать на примере", use_container_width=True,
                 help="Две скважины, которых модель не видела при обучении, вместе с разметкой по керну"):
        st.session_state["example"] = True
    if upload is not None:
        st.session_state["example"] = False

    st.header("2. Порог уверенности")
    threshold = st.slider("Интервалы с уверенностью ниже порога подсвечиваются красным",
                          0.3, 0.9, 0.5, 0.05)

    st.header("Фации")
    st.markdown(legend_html(), unsafe_allow_html=True)

st.title("🪨 Литология по данным ГИС")

raw, source = None, None
if upload is not None:
    try:
        raw, source = read_file(upload.name, upload.getvalue()), upload.name
    except Exception as e:
        st.error(f"Не удалось прочитать файл «{upload.name}»: {e}")
        st.stop()
elif st.session_state.get("example"):
    full = pd.read_csv(EXAMPLE_PATH)
    raw = full[full["Well Name"].isin(info["holdout"]["holdout_wells"])].copy()
    source = "пример: " + ", ".join(info["holdout"]["holdout_wells"])

tab_main, tab_model, tab_help = st.tabs(["📈 Результат", "🧠 О модели", "❓ Как подготовить файл"])

with tab_help:
    st.markdown("Загрузите таблицу, где **одна строка — одна глубина**. Названия столбцов могут отличаться: "
                "сайт попробует узнать их сам, а если не узнает — вы выберете вручную.")
    need = pd.DataFrame({
        "Столбец": ["Well Name", "Depth"] + FEATURES + ["Facies", "Formation"],
        "Что это": ["Название скважины", "Глубина, м"] + [FEATURE_HELP[f] for f in FEATURES]
                   + ["Фация по керну, 1–9", "Пласт / свита"],
        "Обязателен": ["нет (если одна скважина)", "да"] + ["да"] * 5
                      + ["да (или Formation, тогда посчитаем)", "да"] + ["нет — для сравнения с моделью", "нет"],
    })
    st.dataframe(need, hide_index=True, use_container_width=True)
    st.markdown("- Сопротивление можно дать в Ом·м (столбец `ILD`, `RT`) — логарифм посчитается сам.\n"
                "- Если есть столбец `Facies` (описание керна), сайт сравнит с ним прогноз.\n"
                "- Строки с пропусками в кривых пропускаются, об этом будет сообщение.")
    template = pd.read_csv(ROOT / "data" / "new_well.csv").head(20)
    st.download_button("Скачать шаблон CSV", template.to_csv(index=False).encode("utf-8"),
                       "шаблон_скважины.csv", "text/csv")

with tab_model:
    h = info["holdout"]
    st.markdown(f"**CatBoost**, обучен на {len(info['train_wells'])} скважинах "
                f"({info['training_rows']} интервалов). Проверка — на скважинах, которых модель не видела: "
                f"{', '.join(h['holdout_wells'])} ({h['holdout_rows']} интервалов).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Точное совпадение с керном", f"{h['accuracy']:.0%}")
    c2.metric("Совпадение с точностью до соседней фации", f"{h['accuracy_adjacent']:.0%}",
              help="Соседние фации геологически близки (например, крупно- и мелкозернистый алевролит)")
    c3.metric("Macro F1", f"{h['macro_f1']:.2f}", help="Среднее качество по всем 9 фациям одинаково — "
                                                        "редкие фации весят столько же, сколько частые")
    left, right = st.columns(2)
    left.plotly_chart(confusion_fig(h["confusion_matrix"], "Какие фации модель путает"),
                      use_container_width=True)
    imp = pd.Series(info["feature_importance"]).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=[f"{k} — {FEATURE_HELP[k]}" for k in imp.index],
                           orientation="h", marker_color="#1565C0"))
    fig.update_layout(title="Какие кривые важнее для модели", height=460, template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10))
    right.plotly_chart(fig, use_container_width=True)
    rec = pd.DataFrame({"Фация": [LABELS[int(k)] for k in h["recall"]],
                        "Доля найденных": [f"{v:.0%}" for v in h["recall"].values()]})
    st.markdown("**Насколько хорошо модель находит каждую фацию** (на проверочных скважинах)")
    st.dataframe(rec, hide_index=True, use_container_width=True)

with tab_main:
    if raw is None:
        st.info("👈 Загрузите файл скважины слева или нажмите **«Попробовать на примере»**.")
        st.markdown(
            "**Что делает сайт**\n\n"
            "1. Берёт каротажные кривые (GR, сопротивление, пористость, PE).\n"
            "2. Для каждой глубины определяет одну из 9 фаций и показывает, насколько модель уверена.\n"
            "3. Рисует планшет: кривые, прогноз и, если есть, разметку по керну рядом.\n"
            "4. Сомнительные интервалы подсвечены — их можно исправить и скачать итоговую разметку.")
        st.stop()

    st.caption(f"Источник: {source} · строк: {len(raw)}")

    # --- сопоставление столбцов ---
    auto, raw_res = guess_mapping(raw.columns)
    missing_auto = [c for c in REQUIRED if auto[c] == NONE]
    with st.expander("Сопоставление столбцов" + (" — нужно проверить ⚠️" if missing_auto else " ✓"),
                     expanded=bool(missing_auto)):
        options = [NONE] + list(raw.columns)
        mapping = {}
        cols = st.columns(4)
        for i, target in enumerate(["Well Name", "Depth"] + FEATURES + ["Facies", "Formation"]):
            hint = FEATURE_HELP.get(target, {"Well Name": "Скважина", "Depth": "Глубина",
                                             "Facies": "Фация по керну", "Formation": "Пласт"}.get(target))
            mapping[target] = cols[i % 4].selectbox(f"{target}", options, index=options.index(auto[target]),
                                                    help=hint, key=f"map_{target}")
        log_res = st.checkbox("Сопротивление дано в Ом·м — взять log₁₀", value=raw_res)

    df = pd.DataFrame(index=raw.index)
    for target, col in mapping.items():
        if col != NONE:
            df[target] = raw[col]
    if "Well Name" not in df:
        df["Well Name"] = Path(source).stem if upload is not None else "Скважина"
    for c in [c for c in REQUIRED + ["Facies"] if c in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if log_res and "ILD_log10" in df:
        df["ILD_log10"] = np.log10(df["ILD_log10"].where(df["ILD_log10"] > 0))
    if "RELPOS" not in df and "Formation" in df and "Depth" in df:
        df["RELPOS"] = add_relpos(df)
        st.info("RELPOS посчитан по кровле и подошве пластов из столбца Formation.")

    missing = [c for c in REQUIRED if c not in df]
    if missing:
        st.error("Не хватает столбцов: " + ", ".join(f"**{c}** ({FEATURE_HELP.get(c, 'глубина')})"
                                                    for c in missing)
                 + ". Укажите их в «Сопоставлении столбцов» выше или добавьте в файл.")
        st.stop()

    bad = df[REQUIRED].isna().any(axis=1)
    if bad.any():
        st.warning(f"Пропущено {bad.sum()} строк с пустыми или нечисловыми значениями кривых.")
        df = df[~bad]
    if df.empty:
        st.error("После проверки не осталось ни одной строки с полными данными.")
        st.stop()
    has_truth = "Facies" in df and df["Facies"].between(1, 9).all()

    result = predict(df, model)

    wells = sorted(result["Well Name"].astype(str).unique())
    well = st.selectbox("Скважина", wells) if len(wells) > 1 else wells[0]
    w = result[result["Well Name"].astype(str) == well].sort_values("Depth")

    m = st.columns(5 if has_truth else 4)
    m[0].metric("Глубины, м", f"{w['Depth'].min():.0f}–{w['Depth'].max():.0f}")
    m[1].metric("Точек", len(w))
    m[2].metric("Средняя уверенность", f"{w['Confidence'].mean():.0%}")
    m[3].metric("Сомнительных", f"{(w['Confidence'] < threshold).mean():.0%}",
                help=f"Доля глубин с уверенностью ниже {threshold:.0%}")
    if has_truth:
        exact = (w["Facies"] == w["Predicted_Facies"]).mean()
        near = np.mean([p == t or p in ADJACENT[int(t)] for t, p in zip(w["Facies"], w["Predicted_Facies"])])
        m[4].metric("Совпадение с керном", f"{exact:.0%}", f"с соседними: {near:.0%}", delta_color="off")

    st.plotly_chart(log_plot(w, threshold, has_truth), use_container_width=True)

    st.subheader("Интервалы")
    st.caption("Подряд идущие глубины с одной фацией объединены. Не согласны с моделью — "
               "выберите фацию в столбце «Решение геолога», правки попадут в скачанный файл.")
    iv = intervals(w, threshold)
    edited = st.data_editor(
        iv, hide_index=True, use_container_width=True, key=f"edit_{source}_{well}",
        disabled=[c for c in iv.columns if c != "Решение геолога"],
        column_config={
            "Решение геолога": st.column_config.SelectboxColumn(options=list(LABELS.values()), required=True),
            "Уверенность": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        })
    changed = (edited["Решение геолога"] != edited["Прогноз"]).sum()

    # правки по интервалам → по глубинам
    label_to_id = {v: k for k, v in LABELS.items()}
    out = w.copy()
    out["Geologist_Facies"] = out["Predicted_Facies"]
    for _, r in edited.iterrows():
        sel = (out["Depth"] >= r["Кровля, м"]) & (out["Depth"] < r["Подошва, м"])
        out.loc[sel, "Geologist_Facies"] = label_to_id[r["Решение геолога"]]
    out["Doubtful"] = out["Confidence"] < threshold

    c1, c2 = st.columns(2)
    c1.download_button(f"⬇️ Скачать результат по скважине {well}" + (f" (правок: {changed})" if changed else ""),
                       out.to_csv(index=False).encode("utf-8"), f"{well}_фации.csv", "text/csv",
                       use_container_width=True)
    c2.download_button("⬇️ Скачать интервалы", edited.to_csv(index=False).encode("utf-8"),
                       f"{well}_интервалы.csv", "text/csv", use_container_width=True)

    if has_truth:
        st.subheader("Сравнение с керном")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(w["Facies"].astype(int), w["Predicted_Facies"], labels=list(range(1, 10)))
        st.plotly_chart(confusion_fig(cm, f"Где модель ошибается на скважине {well}"),
                        use_container_width=True)
