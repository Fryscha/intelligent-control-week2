import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

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

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi ke RGB

    # Normalisasi seluruh frame
    frame_scaled = scaler.transform(frame_rgb.reshape(-1, 3))

    # Prediksi warna untuk seluruh frame
    color_probs = model.predict_proba(frame_scaled)

    # Hitung persentase setiap warna
    color_counts = np.sum(color_probs, axis=0)
    total_pixels = frame.shape[0] * frame.shape[1]
    color_percentages = (color_counts / total_pixels) * 100

    # List untuk menyimpan bounding boxes dengan confidence score
    bounding_boxes = []

    # Tampilkan hasil dan gambar bounding box
    for i, color in enumerate(model.classes_):
        conf = color_percentages[i]
        if conf > 0:  # Hanya tampilkan warna yang terdeteksi
            # Buat mask untuk warna yang terdeteksi
            mask = (color_probs[:, i] > 0.5).reshape(frame.shape[0], frame.shape[1])  # Threshold
            if np.any(mask):
                # Temukan kontur dari mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Filter kontur kecil
                        x, y, w, h = cv2.boundingRect(contour)
                        bounding_boxes.append((x, y, w, h, conf, color))  # Simpan bounding box dengan confidence score

    # Sortir bounding boxes berdasarkan confidence score (descending)
    bounding_boxes.sort(key=lambda x: x[4], reverse=True)

    # Pilih hanya top 2 bounding boxes
    top_2_bounding_boxes = bounding_boxes[:2]

    # Gambar hanya top 2 bounding boxes
    for x, y, w, h, conf, color in top_2_bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Gambar bounding box
        label = f"{color}: {conf:.2f}%"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame dengan bounding box
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
