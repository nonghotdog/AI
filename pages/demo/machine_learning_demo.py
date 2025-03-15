import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดล
model_rf = joblib.load("model\model_RandomForestClassifier.pkl")
model_lr = joblib.load("model\model_LogisticRegression.pkl")


st.write("# Osteoporosis Predictor")

def user_input_features():
    st.sidebar.write("## รายละเอียดบุคคล")
    age = st.sidebar.slider("Age", 20, 100, 50, help="Age of the individual in years.")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Biological sex of the individual.")
    race = st.sidebar.selectbox("Race/Ethnicity", ["Caucasian", "African American", "Asian"], help="Racial or ethnic background.")

    st.sidebar.write("## พฤติกรรมการใช้ชีวิต")
    hormonal_changes = st.sidebar.selectbox("Hormonal Changes", ["Normal", "Postmenopausal"], help="Hormonal status, especially relevant for women.")
    family_history = st.sidebar.selectbox("Family History", ["No", "Yes"], help="History of osteoporosis in the family.")
    body_weight = st.sidebar.selectbox("Body Weight", ["Normal", "Underweight"], help="Body weight category.")
    calcium_intake = st.sidebar.selectbox("Calcium Intake", ["Low", "Adequate"], help="Daily calcium consumption.")
    vitamin_d_intake = st.sidebar.selectbox("Vitamin D Intake", ["Insufficient", "Sufficient"], help="Daily vitamin D intake.")
    physical_activity = st.sidebar.selectbox("Physical Activity", ["Sedentary", "Active"], help="Level of physical activity.")
    smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"], help="Current smoking status.")
    alcohol = st.sidebar.selectbox("Alcohol Consumption", ["No", "Moderate"], help="Alcohol consumption level.")

    st.sidebar.write("## ประวัติการใช้ยา")
    medical_conditions = st.sidebar.selectbox("Medical Conditions", ["No", "Rheumatoid Arthritis"], help="Presence of relevant medical conditions.")
    medications = st.sidebar.selectbox("Medications", ["No", "Corticosteroids"], help="Use of medications known to affect bone health.")
    prior_fractures = st.sidebar.selectbox("Prior Fractures", ["No", "Yes"], help="History of bone fractures.")

    data = pd.DataFrame([[age, gender, hormonal_changes, family_history, race, body_weight, calcium_intake,
                          vitamin_d_intake, physical_activity, smoking, alcohol, medical_conditions, medications,
                          prior_fractures]],
                         columns=["Age", "Gender", "Hormonal Changes", "Family History", "Race/Ethnicity",
                                  "Body Weight", "Calcium Intake", "Vitamin D Intake", "Physical Activity",
                                  "Smoking", "Alcohol Consumption", "Medical Conditions", "Medications",
                                  "Prior Fractures"])
    return data

input_data = user_input_features()

result_rf = model_rf.predict(input_data)[0]
result_lr = model_lr.predict(input_data)[0]

L, R = st.columns(2, border=True)
with L:
    st.write("## :orange[Random Forest]")
    if result_rf == 1:
        st.error("มีความเสี่ยงที่จะเป็นโรค \"กระดูกพรุน\" สูง")
    else:
        st.success("ไม่มี||มีความเสี่ยงที่จะเป็นโรค \"กระดูกพรุน\" ต่ำ")

with R:
    st.write("## :orange[Logistic Regression]")
    if result_lr == 1:
        st.error("มีความเสี่ยงที่จะเป็นโรค \"กระดูกพรุน\" สูง")
    else:
        st.success("ไม่มี||มีความเสี่ยงที่จะเป็นโรค \"กระดูกพรุน\" ต่ำ")
st.divider()

a, b, c = st.columns(3)
a.write("#### รายละเอียดบุคคล")
b.write("#### พฤติกรรมการใช้ชีวิต")
c.write("#### ประวัติการใช้ยา")
d, e, f = st.columns(3, border=True)
with d:
    st.metric(label="อายุ", value=input_data['Age'].values[0])
    st.divider()
    st.metric(label="เพศ", value=input_data['Gender'].values[0])
    st.divider()
    st.metric(label="เชื้อชาติ", value=input_data['Race/Ethnicity'].values[0])
with e:
    st.metric(label="ฮอร์โมน", value=input_data['Hormonal Changes'].values[0])
    st.divider()
    st.metric(label="ประวัติครอบครัว", value=input_data['Family History'].values[0])
    st.divider()
    st.metric(label="น้ำหนัก", value=input_data['Body Weight'].values[0])
    st.divider()
    st.metric(label="แคลเซียม", value=input_data['Calcium Intake'].values[0])
    st.divider()
    st.metric(label="วิตามินดี", value=input_data['Vitamin D Intake'].values[0])
    st.divider()
    st.metric(label="ออกกำลังกาย", value=input_data['Physical Activity'].values[0])
    st.divider()
    st.metric(label="สูบบุหรี่", value=input_data['Smoking'].values[0])
    st.divider()
    st.metric(label="แอลกอฮอล์", value=input_data['Alcohol Consumption'].values[0])
with f:
    st.metric(label="โรคประจำตัว", value=input_data['Medical Conditions'].values[0])
    st.divider()
    st.metric(label="ยาที่ใช้", value=input_data['Medications'].values[0])
    st.divider()
    st.metric(label="กระดูกหักก่อนหน้านี้", value=input_data['Prior Fractures'].values[0])
