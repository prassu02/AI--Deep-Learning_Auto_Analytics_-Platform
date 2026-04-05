# ======================================================
# 🚀 AI AUTO ANALYTICS + DL PLATFORM
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re, csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(layout="wide")
st.title("🚀 AI Deep Learning Auto Analytics Platform")

if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV / Excel / TXT", type=["csv","xlsx","txt"])

if file:

    # =============================
    # LOAD DATA
    # =============================

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)

    else:
        sample = file.read(1024).decode("utf-8")
        file.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        df = pd.read_csv(file, delimiter=dialect.delimiter)

    st.success("✅ Data Loaded")
    st.dataframe(df.head())

    # =============================
    # CLEANING
    # =============================
    for col in df.columns:
        try: 
           df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
        if pd.api.types.is_numeric_dtype(df[col]):
           df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # =============================
    # DASHBOARD
    # =============================

    st.subheader("📊 Auto Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        x = st.selectbox("Feature", df.columns)
        st.plotly_chart(px.histogram(df, x=x))

    with col2:
        y = st.selectbox("Target View", df.columns)
        st.plotly_chart(px.box(df, y=y))

    # =============================
    # NLP MODULE
    # =============================

    text_cols = df.select_dtypes(include="object").columns

    if len(text_cols) > 0:

        st.subheader("🧠 NLP Analysis")

        text_col = st.selectbox("Text Column", text_cols)

        def clean(t):
            return re.sub(r'[^a-zA-Z ]','',str(t).lower())

        df["clean_text"] = df[text_col].apply(clean)

        tfidf = TfidfVectorizer(max_features=100)
        X_text = tfidf.fit_transform(df["clean_text"]).toarray()

        st.write("TF-IDF Shape:", X_text.shape)

    # =============================
    # AUTOML + DL
    # =============================

    st.subheader("🤖 AutoML + Deep Learning")

    target = st.selectbox("Select Target", df.columns)
    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    # Handle infinity
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values
    X = X.fillna(0)

    # Scale
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if y.dtype == "object":
        task = "classification"
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    else:
        task = "regression"

    # Base Model
    base_model = RandomForestClassifier() if task=="classification" else RandomForestRegressor()
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)

    base_score = accuracy_score(y_test, base_pred) if task=="classification" else r2_score(y_test, base_pred)

    # Deep Learning Model (MLP)
    if task == "classification":
        dl_model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200)
    else:
        dl_model = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=200)

    dl_model.fit(X_train, y_train)
    dl_pred = dl_model.predict(X_test)

    dl_score = accuracy_score(y_test, dl_pred) if task=="classification" else r2_score(y_test, dl_pred)

    # =============================
    # RESULTS
    # =============================

    st.subheader("🏆 Model Results")

    st.write(f"RandomForest Score: {base_score}")
    st.write(f"Deep Learning (MLP) Score: {dl_score}")

    comp_df = pd.DataFrame({
        "Model": ["RandomForest", "DeepLearning"],
        "Score": [base_score, dl_score]
    })

    st.plotly_chart(px.bar(comp_df, x="Model", y="Score"))

    # =============================
    # TRACKING
    # =============================

    st.session_state.history.append({
        "RF": base_score,
        "DL": dl_score
    })

    st.subheader("📈 Model Tracking")
    hist_df = pd.DataFrame(st.session_state.history)
    st.line_chart(hist_df)

else:
    st.info("Upload dataset to start")
