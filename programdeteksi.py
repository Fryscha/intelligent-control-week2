import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from scipy.special import softmax

# Path ke dataset
dataset_folder = 'training_dataset'

# List untuk menyimpan data
data = []
labels = []

# Looping ke setiap folder warna
for color_name in os.listdir(dataset_folder):
    color_path = os.path.join(dataset_folder, color_name)

    if os.path.isdir(color_path):  # Pastikan hanya membaca folder
        for img_name in os.listdir(color_path):
            img_path = os.path.join(color_path, img_name)

            # Baca gambar dalam format BGR dan konversi ke RGB
            img = cv2.imread(img_path)
            if img is None:
                continue  # Lewati file yang tidak bisa dibaca
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # FIXED: Konversi ke RGB

            # Ambil rata-rata nilai warna (R, G, B)
            avg_color = img.mean(axis=(0, 1))  # Mean diambil dari semua piksel

            # Simpan fitur dan label
            data.append(avg_color)
            labels.append(color_name)

# Konversi ke array numpy
X = np.array(data)
y = np.array(labels)

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Pilih model SVM
model = SVC(kernel='linear', random_state=42, probability=True)  # Mengatur probability=True
model.fit(X_train, y_train)

# Evaluasi model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Akurasi pada data latih: {train_acc * 100:.2f}%")
print(f"Akurasi pada data uji: {test_acc * 100:.2f}%")

# Simpan model dan scaler
joblib.dump(model, 'color_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model dan scaler berhasil disimpan!")

# Muat kembali model dan scaler untuk deteksi warna
model = joblib.load('color_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # FIXED: Konversi ke RGB

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height // 2, width // 2]

    # Pastikan bentuk data sesuai untuk StandardScaler
    pixel_center_scaled = scaler.transform(pixel_center.reshape(1, -1))

    # Prediksi warna dan skor kepercayaan
    color_pred = model.predict(pixel_center_scaled)[0]
    conf_scores = model.predict_proba(pixel_center_scaled)[0]
    conf = conf_scores[np.argmax(conf_scores)]

    # Tampilkan warna dan skor kepercayaan pada frame
    label = f"{color_pred}: {conf:.2f}"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Konversi kembali ke BGR untuk tampilan OpenCV
    cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
