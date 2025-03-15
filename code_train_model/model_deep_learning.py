import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

train_dir = 'datasets\Training Data\Training Data'
validation_dir = 'datasets\Validation Data\Validation Data'
test_dir = 'datasets\Testing Data\Testing Data'

train_datagen = ImageDataGenerator( # อ่านหนังสือ
    rescale=1./255,           # range pixel 0-1
    rotation_range=30,        # range หมุนๆ
    width_shift_range=0.2,    # range y
    height_shift_range=0.2,   # range x
    shear_range=0.2,          # range บิดภาพ
    zoom_range=0.2,           # range zoom
    horizontal_flip=True,     # เปิดกลับ imaghe
    fill_mode='nearest'       # เติม pixel
)

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

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False # ไม่ train base_model layers (frozen)

model = Sequential()
model.add(base_model)
# convolutional => 1D
model.add(Flatten())

# mid
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# output
model.add(Dense(15, activation='softmax'))

# train
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
checkPoint = ModelCheckpoint('savePointAnimals.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkPoint]
)
model.load_weights('trained_model_file\savePointAnimals.h5')

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples//test_generator.batch_size)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

model.save('trained_model_file\animal_classification_model.h5')
print(".file model แย้ว!")