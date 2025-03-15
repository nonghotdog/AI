import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("datasets\pukpik.csv")

X = df.drop(columns=["Id", "Osteoporosis"]) # drop col ม่ใช้ทิ้ง
y = df["Osteoporosis"] # จะ predict โรค

numeric_features = ["Age"]
categorical_features = ["Gender", "Hormonal Changes", "Family History", "Race/Ethnicity", "Body Weight", "Calcium Intake", "Vitamin D Intake", "Physical Activity", "Smoking", "Alcohol Consumption", "Medical Conditions", "Medications", "Prior Fractures"]

preprocessor = ColumnTransformer(transformers=[ # แบ่งประเภท Data
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ranFor
rf_pipeline = Pipeline([ # ใช้ Pipeline แปลง data
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# regress
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)

joblib.dump(rf_pipeline, "trained_model_file\model_RandomForestClassifier.pkl")
joblib.dump(lr_pipeline, "trained_model_file\model_LogisticRegression.pkl")