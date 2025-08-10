# src/predict_video.py

import pandas as pd
import joblib

# تحميل الموديل المدرب
model = joblib.load("models/parkinson_model.pkl")

# تحميل المميزات المستخرجة للفيديو الجديد
df = pd.read_csv("features/test_features.csv")

# التأكد من الأعمدة المطلوبة
X = df[["avg_stride_length", "total_frames"]]

# التنبؤ
prediction = model.predict(X)[0]

# عرض النتيجة
if prediction == 1:
    print("🧠 PARKINSON DISEASED")
else:
    print("🧠 normal person")
