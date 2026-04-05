# ======================================================
# 🚀 ENTERPRISE AI- DL PLATFORM-------------- (DL + AutoML + NLP + Tracking)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import torch
import re, csv, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(layout="wide")
st.title("🚀 Enterprise AI- DL Analytics Platform")

# ======================================================
# SESSION STATE (MODEL TRACKING)
# ======================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload Dataset", type=["csv","xlsx","txt"])

if file:

    # =============================
    # LOAD DATA
    # =============================

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)

    elif file.name.endswith(".txt"):
        sample = file.read(1024).decode("utf-8")
        file.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        df = pd.read_csv(file, delimiter=dialect.delimiter)

    st.success("✅ Data Loaded")
    st.dataframe(df.head())

    # =============================
    # CLEANING
    # =============================

    df = df.drop_duplicates()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df = df.sample(min(len(df), 4000))

    # =============================
    # DASHBOARD
    # =============================

    st.subheader("📊 Analytics Dashboard")

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

        st.subheader("🧠 NLP Engine")

        text_col = st.selectbox("Text Column", text_cols)

        def clean(t):
            return re.sub(r'[^a-zA-Z ]','',str(t).lower())

        df["clean_text"] = df[text_col].apply(clean)

        tfidf = TfidfVectorizer(max_features=100)
        X_text = tfidf.fit_transform(df["clean_text"]).toarray()

        st.write("TF-IDF:", X_text.shape)

    # =============================
    # AUTOML + DL
    # =============================

    st.subheader("🤖 AutoML + Deep Learning Engine")

    target = st.selectbox("Select Target", df.columns)

    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    X = StandardScaler().fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # =============================
    # TASK TYPE
    # =============================

    if y.dtype == "object":
        task = "classification"
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    else:
        task = "regression"

    # =============================
    # BASE MODEL
    # =============================

    base_model = RandomForestClassifier() if task=="classification" else RandomForestRegressor()
    base_model.fit(X_train,y_train)

    base_pred = base_model.predict(X_test)

    base_score = accuracy_score(y_test,base_pred) if task=="classification" else r2_score(y_test,base_pred)

    # =============================
    # ADVANCED DL MODEL
    # =============================

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
    ])

    if task == "classification":
        model.add(tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax'))
        loss = "sparse_categorical_crossentropy"
    else:
        model.add(tf.keras.layers.Dense(1))
        loss = "mse"

    model.compile(optimizer='adam', loss=loss)

    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    dl_pred = model.predict(X_test)

    if task == "classification":
        dl_pred = np.argmax(dl_pred, axis=1)
        dl_score = accuracy_score(y_test, dl_pred)
    else:
        dl_score = r2_score(y_test, dl_pred)

    # =============================
    # MODEL TRACKING
    # =============================

    st.session_state.history.append({
        "Baseline": base_score,
        "DeepLearning": dl_score
    })

    # =============================
    # RESULTS
    # =============================

    st.subheader("🏆 Model Results")

    st.write(f"Baseline Score: {base_score}")
    st.write(f"Deep Learning Score: {dl_score}")

    comp_df = pd.DataFrame({
        "Model":["Baseline","Deep Learning"],
        "Score":[base_score, dl_score]
    })

    st.plotly_chart(px.bar(comp_df, x="Model", y="Score"))

    # =============================
    # MODEL HISTORY
    # =============================

    st.subheader("📈 Model Tracking")

    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    st.line_chart(hist_df)

else:
    st.info("Upload dataset to begin")
