import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from data import BANDS, CLASSES, COLORS, scene, training_samples

st.set_page_config(page_title="Invasive Species Mapping", page_icon="🌲", layout="wide")
st.title("🌲 Invasive Alien Tree Mapping – Overberg, South Africa")
st.caption("Supervised classification of Sentinel-2 bands (10 m) to map pine, eucalyptus and acacia invasions · simulated reflectance and scene")
FEATS = BANDS + ["NDVI"]
NAMES = list(CLASSES)


@st.cache_resource
def train():
    df = training_samples()
    Xtr, Xte, ytr, yte = train_test_split(df[FEATS], df.label, test_size=.3, random_state=0, stratify=df.label)
    models = {"Random forest": RandomForestClassifier(400, random_state=0, n_jobs=-1).fit(Xtr, ytr),
              "Gradient boosting": HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)}
    return df, Xte, yte, models


@st.cache_data
def classify(model_name):
    truth, X = scene()
    pred = models[model_name].predict(X[FEATS])
    return truth, np.array([NAMES.index(p) for p in pred]).reshape(truth.shape)


df, Xte, yte, models = train()
with st.sidebar:
    model_name = st.radio("Classifier", list(models))
    st.markdown("Pixel = 10 m × 10 m (0.01 ha) · Scene 120 × 120 px = 144 ha")
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")

m = models[model_name]
pred = m.predict(Xte)
k = st.columns(4)
k[0].metric("Overall accuracy", f"{accuracy_score(yte, pred):.1%}")
k[1].metric("Cohen's kappa", f"{cohen_kappa_score(yte, pred):.3f}")
inv = [c for c in NAMES if "invasive" in c]
k[2].metric("F1 – invasive classes", f"{f1_score(yte, pred, labels=inv, average='macro'):.3f}")
k[3].metric("Training pixels", f"{len(df):,}")

t1, t2, t3 = st.tabs(["Classified map", "Accuracy assessment", "Spectral signatures"])
with t1:
    truth, cls = classify(model_name)
    scale = [[i / (len(NAMES) - 1), COLORS[n]] for i, n in enumerate(NAMES)]
    a, b = st.columns([3, 2])
    fig = go.Figure(go.Heatmap(z=cls, colorscale=scale, zmin=0, zmax=len(NAMES) - 1, showscale=False,
                               customdata=np.array(NAMES)[cls], hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(title=f"Classified scene – {model_name}", height=560, yaxis=dict(scaleanchor="x", autorange="reversed"))
    a.plotly_chart(fig, use_container_width=True)
    area = pd.Series(cls.ravel()).map(dict(enumerate(NAMES))).value_counts().reindex(NAMES).fillna(0) * 0.01
    b.plotly_chart(px.bar(area, orientation="h", color=area.index, color_discrete_map=COLORS, labels={"value": "Area (ha)", "index": ""},
                          title="Mapped area by class").update_layout(showlegend=False), use_container_width=True)
    b.metric("Invaded area", f"{area[inv].sum():,.1f} ha", f"{area[inv].sum() / area.sum():.1%} of scene", delta_color="off")
    b.metric("Map agreement with reference", f"{(truth == cls).mean():.1%}")
with t2:
    cm = confusion_matrix(yte, pred, labels=NAMES)
    st.plotly_chart(px.imshow(cm, x=NAMES, y=NAMES, text_auto=True, color_continuous_scale="Blues",
                              labels=dict(x="Predicted", y="Reference"), title="Confusion matrix (test pixels)", height=560), use_container_width=True)
    pa = cm.diagonal() / cm.sum(1)
    ua = cm.diagonal() / cm.sum(0)
    st.dataframe(pd.DataFrame({"Producer's accuracy": pa, "User's accuracy": ua}, index=NAMES).style.format("{:.1%}"), use_container_width=True)
    if model_name == "Random forest":
        st.plotly_chart(px.bar(pd.Series(m.feature_importances_, index=FEATS).sort_values(), orientation="h",
                               title="Band importance", labels={"value": "Importance", "index": ""}).update_layout(showlegend=False), use_container_width=True)
with t3:
    wl = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
    sig = df.groupby("label")[BANDS].mean().T
    sig.index = wl
    st.plotly_chart(px.line(sig, markers=True, color_discrete_map=COLORS, labels={"index": "Wavelength (nm)", "value": "Reflectance (×10⁻⁴)", "label": "Class"},
                            title="Mean spectral signatures"), use_container_width=True)
