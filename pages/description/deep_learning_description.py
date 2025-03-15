import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write('''
         # Neuron Network :orange[Animal Species Classifier] :blue[[อธิบาย]]''')
st.divider()

st.write('''## คือ :orange[Model] อะไร ?''')
q,w,e = st.columns(3)
w.image("graphic/pic/tenor.gif")
st.write('''### :orange[Animal Species Classifier] ถูกพัฒนามาเพื่อ:orange[จำแนก]ชนิดของสัตว์ โดยมีทั้งหมด :blue[15] ชนิด''')
i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 = st.columns(15)
i1.image('https://www.field-studies-council.org/wp-content/uploads/2019/09/beetle.jpg', caption="Beetle")
i2.image('https://upload.wikimedia.org/wikipedia/commons/1/1e/Monarch_butterfly_in_Grand_Canary.jpg', caption="Butterfly")
i3.image('https://static.streamlit.io/examples/cat.jpg', caption="Cat")
i4.image('https://thepetwiki.com/wp-content/uploads/300px-Cow.jpg', caption="Cow")
i5.image('https://static.streamlit.io/examples/dog.jpg', caption="Dog")
i6.image('https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg', caption="Elephant")
i7.image('https://upload.wikimedia.org/wikipedia/commons/b/bb/Gorille_des_plaines_de_l%27ouest_%C3%A0_l%27Espace_Zoologique.jpg', caption="Gorilla")
i8.image('https://static.bangkokpost.com/media/content/dcx/2024/09/21/5278757_790.jpeg', caption="Hippo")
i9.image('https://upload.wikimedia.org/wikipedia/commons/c/c3/Phelsuma_l._laticauda.jpg', caption="Lizard")
i10.image('https://upload.wikimedia.org/wikipedia/commons/4/43/Bonnet_macaque_%28Macaca_radiata%29_Photograph_By_Shantanu_Kuveskar.jpg', caption="Monkey")
i11.image('https://i.pinimg.com/736x/5e/34/26/5e342664134995cb79a4b2069d1a3419.jpg', caption="Mouse")
i12.image('https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/0edd837b-eb3c-42fd-8f32-9a04c89d1619/d9ajd5c-3ceb07a3-1d0e-4f51-9e3b-2c29d2ef6a15.jpg/v1/fill/w_756,h_1056,q_70,strp/kung_fu_panda_3__hi_res_textless_poster__by_phetvanburton_d9ajd5c-pre.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTQzMCIsInBhdGgiOiJcL2ZcLzBlZGQ4MzdiLWViM2MtNDJmZC04ZjMyLTlhMDRjODlkMTYxOVwvZDlhamQ1Yy0zY2ViMDdhMy0xZDBlLTRmNTEtOWUzYi0yYzI5ZDJlZjZhMTUuanBnIiwid2lkdGgiOiI8PTEwMjQifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.4Wh96jK7xJ_oFe9yKQmTPYp5bgB4nKHimHuY5dD57DI', caption="Panda")
i13.image('https://hips.hearstapps.com/hmg-prod/images/jumping-spider-royalty-free-image-1568321050.jpg', caption="Spider")
i14.image('https://www.khaosod.co.th/wpapp/uploads/2024/12/TT02-%E0%B8%A3%E0%B8%B9%E0%B8%9B%E0%B9%80%E0%B8%94%E0%B9%88%E0%B8%99-535x696.jpg', caption="Tiger")
i15.image('https://i.ebayimg.com/images/g/xS0AAOSwN7tgYgzm/s-l1200.jpg', caption="Zebra")
st.divider()

st.write('''
         ## about :blue[Dataset]
         :green[data souce] : ***https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset***
         
         1. ผมจะแบ่ง **Datasets** ออกเป็น :blue[3] part ก็คือ
         - Training     :gray[ส่วนของ data ที่ให้ model เอาไว้เรียนรู้เหมือนกับการ \"ทำแบบฝึกหัด\"]
         - Validation   :gray[\"ฝึกทำข้อสอบ\"]
         - Testing      :gray[\"ทำข้อสอบจริง\"]''')

a,b,c,d = st.columns(4)
a.metric("จำนวนข้อมูลทั้งหมด", "34,000 รูป"," ขนาด 256×256 pixel", border=True, delta_color="off")
b.metric("Training", "3,000 รูป", border=True)
c.metric("Validation ", "2,000 รูป", border=True)
d.metric("Testing ", "2,000 รูป", border=True)

st.write('2. ทำการ Download dataset จาก :blue[Kaggle] ได้ .zip แล้ว extract folder ไปยัง path เดียวกับ .py แล้วทำการ assign ใส่ตัวแปร')
st.code('''
    train_dir = 'datasets\Training Data\Training Data'
    validation_dir = 'datasets\Validation Data\Validation Data'
    test_dir = 'datasets\Testing Data\Testing Data' '''
, language="python")

st.write('''3. กำหนด batch training ปรับ pixel ให้อยู่ใน range 0-1 และเพิ่มตัว augmentation เพื่อความหลากหลาย จะได้ไม่ overfit ดีกับ unseen data''')
st.code('''
    rescale=1./255,           # range pixel 0-1
    rotation_range=30,        # range หมุนๆ
    width_shift_range=0.2,    # range y
    height_shift_range=0.2,   # range x
    shear_range=0.2,          # range บิดภาพ
    zoom_range=0.2,           # range zoom
    horizontal_flip=True,     # เปิดกลับ imaghe
    fill_mode='nearest'       # เติม pixel
)
''', language="python")

st.write('''4. เหมือนกับ train_datagen ทำการ norm แต่ไม่มีตัว augmentation เพราะจะเอามา valid และ test ''')
st.code('''
    validation_datagen = ImageDataGenerator(rescale=1./255) # ทำ pre ข้อสอบ
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    validation_generator = validation_datagen.flow_from_directory( # ทำ pre ข้อสอบ
        validation_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = validation_datagen.flow_from_directory( # สอบจริง
        test_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
''', language="python")
st.divider()

st.write('''
         ## :blue[Training] Model :orange[Algoritms]
         ผมให้ model เรียนรู้แบบ Transfer Learning โดยใช้ **VGG16(CNN)** (เป็นmodelที่เคยฝึกมาก่อนแล้ว) เพราะประหยัดเวลาและทรัพยากรในการ train modle ส่วนใหญ่จะเน้นไปที่การออกแบบตัว augment
        
         # AGG16 ''')
st.image("graphic/pic/vgg16.png", caption="source : ***https://www.cs.toronto.edu/~frossard/post/vgg16/***")
st.write('''
        ตามชื่อเลย AGG16 มี 16 layer ใช้ filter ขนาดเล็ก เช่น 3x3 convolution filters และ stride = 1 เนื่องจากการ ขนาดของ filter ที่เล็ก การทำ padding เลยสำคัญ แต่จำนวน layer มีขนาดลึกเลยเอามาหักล้างหรือทดแทนกันได้ และ max pooling 2x2 และ หลังจากโดน convolution และ pooling ข้อมูลที่ได้จะถูกแปลงเป็น vector ไปยัง fully connected เพื่อ output
        แล้ว AGG16 ก็เหมือนกับ deep learning ตัวอื่นๆ ใช้แนวคิด gradient descent กับ backpropagation 

         ***https://www.youtube.com/watch?v=QW7aygOH22I&ab_channel=Wuttipong%E0%B8%A7%E0%B8%B8%E0%B8%92%E0%B8%B4%E0%B8%9E%E0%B8%87%E0%B8%A9%E0%B9%8CKumwilaisak%E0%B8%84%E0%B9%8D%E0%B8%B2%E0%B8%A7%E0%B8%B4%E0%B8%A5%E0%B8%B1%E0%B8%A2%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%94%E0%B8%B4%E0%B9%8C***''')

st.write('''''')
st.write('''
         1. กำหนดค่า VGG16 ไม่ให้ฝึกใน layer ของตัวเอง''')
st.code('''
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False '''
, language="python")

st.write('''2. ต้องแปลงไปเป็น 1D เพื่อจะได้ไปเข้า dense(Fully Connected) ต่อไปได้''')
st.code('''
    # input
    model = Sequential()
    model.add(base_model)
    model.add(Flatten()) '''
, language="python")

st.write('''3. ออกแบบ layer dense กับ dropout และ กัน vanishing gradient''')
st.code('''
    # hindden
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) '''
, language="python")

st.write('''4. หลังจากนั้นจะใช้ softmax ให้ output ออกมาเป็นค่า prob เพราะมีหลายประเภท''')
st.code('''
    # output
    model.add(Dense(15, activation='softmax')) '''
, language="python")

st.write('''5. ผมเลือกที่จะ step น้อยๆ เพราะไม่อยาก overfit และใช้ adam เป็นตัว optimizer เพราะคิดว่าที่ทำอยู่เหมาะกับการ adaptive learning rate ''')
st.code('''
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy']) '''
, language="python")

st.write('''6. ทำการ callback เพื่อให้ save model ที่ดีที่สุดเก็บไว้ และเผื่อกรณีคอมดับเพราะผม train นานมาก เพราะข้อจำกัดของ hardware กับกลัวไฟไหม้หอ''')
st.code('''
    checkPoint = ModelCheckpoint('savePointAnimals.h5', monitor='val_loss', save_best_only=True) '''
, language="python")

st.write('''7. กำหนด epochs แบบเยอะๆไว้ก่อนถ้าไม่ถึง ก็สามารถใช้ที่ callback ไว้''')
st.code('''
    history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkPoint]
) '''
, language="python")

st.write('''8. mornitor เพื่อดู behavior ของ model''')
st.code('''
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples//test_generator.batch_size)
) '''
, language="python")
st.divider()

st.write("# Timeline Accuracy & Hardware Details")
video_file = open("graphic/vid/envidence.mkv", "rb")
video_bytes = video_file.read()
st.video(video_bytes)
border, = st.columns(1)
border.metric("Model Accuracy", "81.25 %", border=True)
border.metric("Test Accuracy", "79.62 %", "-1.63 %", border=True)