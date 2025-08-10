# src/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# تحميل البيانات
df = pd.read_csv("features/features.csv")

# تأكد من وجود عمود label
if "label" not in df.columns:
    raise ValueError("❌ لازم تضيف عمود 'label' لملف features.csv (1=مريض, 0=سليم)")

# اختيار الخصائص
X = df[["avg_stride_length", "total_frames"]]
y = df["label"]

# بناء الموديل
model = RandomForestClassifier(n_estimators=100, random_state=42)

if len(df) < 3:
    print("⚠️ عدد العينات قليل، هنستخدم كل الداتا في التدريب والاختبار.")
    model.fit(X, y)
    y_pred = model.predict(X)
    print("🔎 تقييم النموذج على نفس البيانات:")
    print(classification_report(y, y_pred))
else:
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("🔎 تقييم النموذج على مجموعة الاختبار:")
    print(classification_report(y_test, y_pred))

# حفظ النموذج
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/parkinson_model.pkl")
print("✅ تم حفظ الموديل في models/parkinson_model.pkl")
