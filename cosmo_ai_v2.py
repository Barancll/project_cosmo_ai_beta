"""
Created on Fri Jul 14 01:47:39 2023

@author: bct
"""
import tensorflow as tf
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Pillow'un uzun görsel yüklemelerini desteklemesi için
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.model_selection import train_test_split

#Open CSV
data = pd.read_csv('cosmo_ai.csv', sep = ';')

# Veri seti dizini ve sınıf etiketleri
data_img = 'C:/Users/bct/Desktop/cosmo_ai'

# ImageDataGenerator ile veri setini yüklemek ve ön işlemek
datagen = ImageDataGenerator(rescale=1./255)

# Tüm veri setini yükleme (train + validation)
data_generator = datagen.flow_from_directory(
    directory=os.path.join(data_img, 'img'),
    target_size=(224, 224),
    batch_size=6,
    class_mode='categorical',
    shuffle=True
)

# CNN modelini oluşturma ve derleme (sınıf sayısını düzeltin)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(data_generator.class_indices), activation='softmax'))  # Sınıf sayısını otomatik olarak alın

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    data_generator,
    steps_per_epoch=data_generator.samples // data_generator.batch_size,
    epochs=10
)

# Görüntüyü tahmin etme
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = list(data_generator.class_indices.keys())[predicted_class]
    return predicted_label

# Test görüntülerinin yolu
test_goruntuleri = ['test_img/test1.jpg']

# Test görüntülerini tahmin etme
for goruntu in test_goruntuleri:
    goruntu_yolu = os.path.join(data_img, goruntu)
    tahmin = predict_image(goruntu_yolu)
    print(f"{goruntu} --> Tahmin: {tahmin}")

# CSV dosyasından gezegenin uzaklık mesafesini okuma
gezegen_mesafe = None

try:
        gezegen_mesafe = data.loc[data['Gezegenler'] == tahmin, 'Uzaklık '].values[0]
except IndexError:
        print(f"Uzaklık Mesafesi bilgisi bulunamadı: {tahmin}")
        
        
# Sonuçları yeni bir CSV dosyasına yazma
if gezegen_mesafe is not None:
        result_df = pd.DataFrame({'Test Görüntüsü': [goruntu], 'Tahmin Edilen Gezegen': [tahmin], 'Uzaklık Mesafesi': [gezegen_mesafe]})
        with open('tahmin_sonuclari.csv', 'a', newline='') as file:
            result_df.to_csv(file, index=False, header=not file.tell())
 
# Orijinal girdi görselini yükleme
girdi_gorsel = Image.open(os.path.join(data_img, test_goruntuleri[0]))

# Tahmin edilen sınıfın görselini yükleme (jpg formatında)
tahmin_gorseli_yolu_jpg = os.path.join(data_img, 'tahmin_img', tahmin.lower() + '.jpg')
tahmin_gorseli_jpg = Image.open(tahmin_gorseli_yolu_jpg)

# Görselleri yan yana görüntüleme
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Orijinal girdi görselini ekranda gösterme
ax[0].imshow(girdi_gorsel)
ax[0].set_title("Test Görseli")
ax[0].axis('off')

# Tahmin edilen sınıfın jpg formatındaki görselini ekranda gösterme
ax[1].imshow(tahmin_gorseli_jpg)
ax[1].set_title(f"Tahmin (JPG): {tahmin}")
ax[1].axis('off')

# Görselleri görüntüleme
plt.show()

