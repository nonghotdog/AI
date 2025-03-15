import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model/savePointAnimals.h5')

class_labels = [
    'Beetle', 'Butterfly', 'Cat', 'Cow', 'Dog', 'Elephant', 'Gorilla', 'Hippo', 'Lizard', 'Monkey', 'Mouse', 'Panda', 'Spider', 'Tiger', 'Zebra'
]

class_to_emoji = {
    'Beetle': '🐞',
    'Butterfly': '🦋',
    'Cat': '🐱',
    'Cow': '🐄',
    'Dog': '🐶',
    'Elephant': '🐘',
    'Gorilla': '🦍',
    'Hippo': '🦛',
    'Lizard': '🦎',
    'Monkey': '🐒',
    'Mouse': '🐭',
    'Panda': '🐼',
    'Spider': '🕷️',
    'Tiger': '🐯',
    'Zebra': '🦓',
}

def predict_image(_img):
    if _img.mode != 'RGB': # ถ้าเป็นสีอื่นๆแปลงเป็น RGB ผมเจอกรณีสี type alpha (RGBA)
        _img = _img.convert('RGB')

    _img = _img.resize((256, 256))  # scale input ให้เข้ากับ model
    img_array = np.array(_img) / 255.0  # [0, 1] pixel
    img_array = np.expand_dims(img_array, axis=0)  # dimension batch

    prediction = model.predict(img_array)  # เอาใส model
    predicted_class_idx = np.argmax(prediction)  # หาค่า max //class ที่เป็นไปได้
    predicted_class = class_labels[predicted_class_idx]  # index(int) class => class
    confidence = prediction[0][predicted_class_idx]  # score

    return predicted_class, confidence

st.write("# Animal Species Classifier")
st.sidebar.write("# อัพโหลดรูปภาพ")
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.image('https://placehold.co/600x200/transparent/FFFFFF?text=Image', caption="ไม่มีรูปภาพ", use_container_width=True)
    st.sidebar.warning('โปรด upload รูปภาพ ⚠️', icon="☝️")
    d, e = st.columns(2)
    d.metric("class", "???", border=True)
    e.metric("confidence", "???", border=True)

else:
    try:
        img = Image.open(uploaded_file)
        a, b, c = st.columns(3)
        with b:
            with st.spinner('model กำลัง predict...'):
                st.image(img, caption="ภาพที่อัปโหลดสัตว์ :)")

        predicted_class, confidence = predict_image(img)
        if confidence < 0.3:
            color = "red"
        elif confidence < 0.5:
            color = "orange"
        else:
            color = "green"

        emoji = class_to_emoji.get(predicted_class,"")

        f, g = st.columns(2)
        f.metric("class", f"{predicted_class} {emoji}", border=True)
        g.metric("confidence", f"{confidence*100}", border=True)
        st.write(f"# 🤖 Model บอกว่าน้องเป็น :green[{predicted_class}] ด้วยความมั่นใจ :{color}[{confidence * 100:.2f}%] {emoji}")
        st.balloons()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {e}")

st.sidebar.write('''## Model รองรับสัตว์ 15 ชนิด           
    - Beetle
    - Butterfly
    - Cat
    - Cow
    - Dog
    - Elephant
    - Gorilla
    - Hippo
    - Lizard
    - Monkey
    - Mouse
    - Panda
    - Spider
    - Tiger
    - Zebra
''')