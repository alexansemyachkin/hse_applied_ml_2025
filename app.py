import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(page_title="Car Price Prediction", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "model.pkl"
VIZ_PREP_PATH = MODEL_DIR / "vizualization_preprocessor.pkl"
FEATURE_NAMES_IN_PATH = MODEL_DIR / "feature_names_in.pkl"
FEATURE_NAMES_OUT_PATH = MODEL_DIR / "feature_names_out.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_viz_preprocessor():
    with open(VIZ_PREP_PATH, "rb") as f:
        return pickle.load(f)
    

@st.cache_resource
def load_feature_names_in():
    with open(FEATURE_NAMES_IN_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_names_out():
    with open(FEATURE_NAMES_OUT_PATH, "rb") as f:
        return pickle.load(f)


def run_eda(df_viz: pd.DataFrame):
    st.header("EDA")

    num_cols = df_viz.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df_viz.select_dtypes(include=["object"]).columns.tolist()

    st.subheader("Распределение числовых признаков")
    for col in num_cols:
        fig = px.histogram(df_viz, x=col, nbins=30, title=col)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Распределение категориальных признаков")
    for col in cat_cols:
        fig = px.histogram(df_viz, x=col, title=col)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Корреляционная матрица")
    corr = df_viz[num_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Корреляции числовых признаков"
    )
    st.plotly_chart(fig, use_container_width=True)


def run_prediction(model, df_raw: pd.DataFrame):
    st.header("Предсказание стоимости автомобиля")

    try:
        preds = model.predict(df_raw)
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        return

    df_pred = df_raw.copy()
    df_pred["predicted_price"] = preds

    st.subheader("Результаты предсказания")
    st.dataframe(df_pred.head())

    st.download_button(
        "Скачать предсказания",
        df_pred.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )


def visualize_model_weights(model, feature_names):
    st.header("Коэффициенты линейной модели Ridge")

    try:
        ridge = model.regressor_.named_steps["regressor"]
        prep = model.regressor_.named_steps["preprocessor"]
        feature_names = feature_names
    except Exception as e:
        st.error(f"Не удалось извлечь веса модели: {e}")
        return

    coefs = (
        pd.DataFrame({"feature": feature_names, "weight": ridge.coef_})
        .sort_values(by="weight", key=np.abs, ascending=False)
    )

    st.dataframe(coefs)

    fig = px.bar(
        coefs,
        x="weight",
        y="feature",
        orientation="h",
        title="Важность признаков"
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Предсказание стоимости автомобиля")

    try:
        model = load_model()
        viz_pre = load_viz_preprocessor()
        feature_names_in = load_feature_names_in()
        feature_names_out = load_feature_names_out()
    except Exception as e:
        st.error(f"Ошибка загрузки модели или препроцессора или признаков: {e}")
        return

    uploaded = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if uploaded is None:
        st.info("Пожалуйста, загрузите CSV-файл")
        return

    df_raw = pd.read_csv(uploaded)
    st.subheader("Исходные данные")
    st.dataframe(df_raw.head())

    try:
        df_viz_pre = viz_pre.fit_transform(df_raw.copy())
        df_viz = pd.DataFrame.from_records(df_viz_pre, columns=feature_names_in)
    except Exception as e:
        st.error(f"Ошибка препроцессинга для EDA: {e}")
        return

    run_eda(df_viz)
    run_prediction(model, df_raw)
    visualize_model_weights(model, feature_names_out)


if __name__ == "__main__":
    main()
