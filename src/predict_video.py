# src/predict_video.py

import pandas as pd
import joblib


model = joblib.load("models/parkinson_model.pkl")


df = pd.read_csv("features/test_features.csv")


X = df[["avg_stride_length", "total_frames"]]


prediction = model.predict(X)[0]


if prediction == 1:
    print("🧠 PARKINSON DISEASED")
else:
    print("🧠 normal person")
