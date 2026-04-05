# ======================================================
# 🚀 PURE DEEP LEARNING PLATFORM (END-TO-END)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import re, csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(layout="wide")
st.title("🚀 Pure Deep Learning AI Platform")

# ======================================================
# FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV / Excel / TXT", type=["csv","xlsx","txt"])

if file:

    # =============================
    # LOAD DATA
    # =============================

    try:
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

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

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

    df = df.sample(min(len(df), 3000))

    # =============================
    # DASHBOARD
    # =============================

    st.subheader("📊 Data Visualization")

    x = st.selectbox("Select Feature", df.columns)
    st.plotly_chart(px.histogram(df, x=x))

    # =============================
    # DL MODEL SETUP
    # =============================

    st.subheader("🧠 Deep Learning Model")

    target = st.selectbox("Select Target Column", df.columns)

    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # =============================
    # TASK TYPE
    # =============================

    if y.dtype == "object":
        task = "classification"
        le = LabelEncoder()
        y = le.fit_transform(y)
        output_units = len(np.unique(y))
        activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    else:
        task = "regression"
        output_units = 1
        activation = "linear"
        loss = "mse"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # =============================
    # MODEL ARCHITECTURE
    # =============================

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(output_units, activation=activation)
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'] if task=="classification" else ['mae'])

    # =============================
    # TRAINING
    # =============================

    epochs = st.slider("Epochs", 5, 50, 10)

    if st.button("🚀 Train Deep Learning Model"):

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        st.success("Training Completed")

        # =============================
        # METRICS VISUALIZATION
        # =============================

        hist_df = pd.DataFrame(history.history)

        st.subheader("📈 Training Performance")

        st.line_chart(hist_df)

        # =============================
        # EVALUATION
        # =============================

        loss_val = model.evaluate(X_test, y_test, verbose=0)

        st.subheader("📊 Model Evaluation")

        if task == "classification":
            st.write(f"Accuracy: {loss_val[1]}")
        else:
            st.write(f"MAE: {loss_val[1]}")

        # =============================
        # PREDICTION
        # =============================

        st.subheader("🔮 Make Prediction")

        sample_input = X_test[:1]
        pred = model.predict(sample_input)

        if task == "classification":
            pred = np.argmax(pred)
            st.write(f"Predicted Class: {pred}")
        else:
            st.write(f"Predicted Value: {pred[0][0]}")

else:
    st.info("Upload dataset to start")
