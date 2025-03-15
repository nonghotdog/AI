import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write('''
         # Machine Learning :orange[Osteoporosis Predictor] :blue[[อธิบาย]]''')
st.divider()

st.write('''## คือ :orange[Model] อะไร ?''')
q,w,e = st.columns(3)
w.image("graphic/pic/ml.gif")
st.write('''### :orange[Osteoporosis Predictor] ถูกพัฒนามาเพื่อ:orange[คาดการความเสี่ยง]ที่จะเป็น :red[\"โรคกระดูกพรุน\"] รายบุคล''')
st.divider()

st.write('''## about :blue[Dataset] ''')
st.markdown('''<div style="text-indent: 60px;">
                    ชุดข้อมูลนี้ให้ข้อมูลที่ครอบคลุมเกี่ยวกับปัจจัยด้านสุขภาพที่มีผลต่อการพัฒนาโรคกระดูกพรุน รวมถึงรายละเอียดทางประชากรศาสตร์ การเลือกดำเนินชีวิต ประวัติทางการแพทย์ และตัวชี้วัดสุขภาพกระดูก โดยมีไว้เพื่ออำนวยความสะดวกในการวิจัยการทำนายโรคกระดูกพรุน ทำให้ model ระบุบุคคลที่มีความเสี่ยงได้จากความสัมพัธู์ของข้อมูลด้วยการวิเคราะห์ปัจจัยต่างๆ เช่น อายุ เพศ การเปลี่ยนแปลงของฮอร์โมน และพฤติกรรมการดำเนินชีวิต สามารถช่วยปรับปรุงการจัดการและการป้องกันโรคกระดูกพรุนได้
                </div>''', unsafe_allow_html=True)
st.write(":green[data souce] : ***https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis***")
st.write("### Feature")

a, b = st.columns([1, 3], border=True) # กำหนดให้มีสองคอลัมน์
with a:
    st.write('''
        - Id
        - Age
        - Gender
        - Hormonal Changes
        - Family History
        - Race/Ethnicity
        - Body Weight
        - Calcium Intake
        - Vitamin D Intake
        - Physical Activity
        - Smoking
        - Alcohol Consumption
        - Medical Conditions
        - Medications
        - Prior Fractures
        - Osteoporosis
    ''')

with b:
    c1, c2 = st.columns(2, border=True)
    with c1:
        st.metric("Feature", "16 columns")
    with c2:
        st.metric("Record", "1958 rows")
    st.write("# Example")
    st.dataframe(pd.read_csv("model/pukpik.csv").head())



st.write('1. ผมเลือก target เป็น "Osteoporosis"(โรคกระดูกพรุน) เพื่อจะทำ model ทำนายว่าบุคคลนั้น "เป็น" หรือ "ไม่เป็น" และทำการ drop feature ที่ไม่มีประโยชน์ออก ')
st.code('''
    X = df.drop(columns=["Id", "Osteoporosis"]) # drop col ม่ใช้ทิ้ง
    y = df["Osteoporosis"] # จะ predict โรค '''
, language="python")

st.write('''2. กำหนดประเภทข้อมูล int, type''')
st.code('''
    numeric_features = ["Age"]
    categorical_features = ["Gender", "Hormonal Changes", "Family History", "Race/Ethnicity", "Body Weight", "Calcium Intake", "Vitamin D Intake", "Physical Activity", "Smoking", "Alcohol Consumption", "Medical Conditions", "Medications", "Prior Fractures"]
)
''', language="python")

st.write('''3. ทำ data preprocessing แปลงข้อมูลด้วย ColumnTransformer ''')
st.code('''
    preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])
''', language="python")

st.write(''' 4. แบ่งข้อมูล dataset 2 ส่วน train(80%) test(20%)
         ''')
st.code('''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
''', language="python")
st.divider()

st.write('''
         ## :blue[Training] Model :orange[Algoritms]
         ผมลองเลือกใช้ Random Forest และ Logistic Regression ในการทำนายโรคกระดูกพรุน ซึ่งความซับซ้อนของปัจจัยเสี่ยงหลายอย่างและมีความสัมพันธ์แบบ non-linear ช่น อายุ เพศ ฮอร์โมน พฤติกรรมการใช้ชีวิต และประวัติทางการแพทย์
        
         # Random Forest 
         มีความสามารถในการจัดการกับข้อมูลที่มีความซับซ้อนและมีความสัมพันธ์แบบ non-linear ได้ดี ทำให้สามารถจับ pattern ต่างๆได้ และยังสามารถช่วยระบุปัจจัยเสี่ยงหลักที่ควรให้ความสำคัญในการป้องกันโรค แต่ถึงจะให้ความแม่นยำสูง แต่การตีความผลลัพธ์จะซับซ้อนกว่า''')
        
st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiN2KPoea9rFZo4nb0SZKrBrEUjNv-xaqB7gF6Htl5lY5AtOmKH1yFalD9Y6XHNNgtUYqsJCPUr-7a4MJIvdcubXogxerrskVqKfQGhKSpUyrnroLhEi6P5vMXqYE22J3_dnLRuWiBv5Nw/s0/Random+Forest+03.gif", caption="source : ***https://blog.tensorflow.org/2021/05/introducing-tensorflow-decision-forests.htm***")
st.write('''
        Random Forest หรือ Decision Trees ที่มีจำนวนมาก แต่ละต้นจะถูกฝึกด้วย data ที่ต่างกันเล็กน้อยจากการ samplig (bootstrap aggregating หรือ bagging) และใช้ features ที่ต่างกันในการตัดสินใจ (random feature selection)
และ ในการ predict โรคกระดูกพรุน Random Forest จะใช้ปัจจัยเสี่ยงต่างๆ เช่น อายุ เพศ ฮอร์โมน พฤติกรรมการใช้ชีวิต และประวัติทางการแพทย์ เป็น feature ในการตัดสินใจใน node ของ tress
โดยแต่ละต้นตัดสินใจจะทำนายว่าผู้ป่วยมีความเสี่ยงเป็นโรคกระดูกพรุนหรือไม่ จากนั้น Random Forest จะนำผลลัพธ์จากการทำนายของต้นไม้ทุกต้นมา vote เพื่อตัดสินใจ''')
st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEigncdlNdHFffOYtoj0SEbPdybNRJwlxBp95GWBmB1JCTk1RghsnMnxFjv-kDIX5BRYN24McGYIiQ3v3jIKIMU9zHV8tXDy4_lfwOMGg0jcy0GF3j6JCnLPhq52PID7TbZvrdWU2iUSuco/s0/pasted+image+0+%252813%2529.png", caption="source : ***https://blog.tensorflow.org/2021/05/introducing-tensorflow-decision-forests.htm***")
st.write('''
         ทำ Pipeline ให้ Random Forest (เพราะ เอาไป transform ,k) โดย Randon Forest ใช้ 100 ต้น''')
st.code('''
    rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train) '''
, language="python")
st.divider()

st.write("# Logistic Regression ") 

st.write('''
        Logistic Regression หรือการใช้ logistic function (หรือ sigmoid function) มาแปลงผลลัพธ์ของการคำนวณแบบ non-linear ให้อยู่ในแบบความน่าจะเป็น (probability) ระหว่าง [0-1]
        ซึ่งในการทำนายโรคกระดูกพรุน Logistic Regression จะใช้ factor ต่างๆ เป็นตัวแปรในการคำนวณเชิงเส้น ซึ่งผลลัพธ์ที่ได้จากการคำนวณเชิงเส้นจะถูกแปลงเป็นความน่าจะเป็นที่ผู้ป่วยจะเป็นโรคกระดูกพรุน''')
st.image("https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe6b35d8d-3396-48a9-b391-7221147127ad_1200x500.gif", caption="source : ***https://www.vizuaranewsletter.com/p/7a3***")

st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/0*lzNKQdsjFVaymJTF.gif", caption="source : ***https://medium.com/@arvindarrives25/logistic-regression-everything-about-it-9bb012cdb79e***")
st.write('''
         ทำ Pipeline ให้ Logistic Regression (เพราะ เอาไป transform มา) โดย max_iter=1000 ทำให้แน่ใจว่ามัน convergence ''')
st.code('''
    lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train) '''
, language="python")