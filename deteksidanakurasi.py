import cv2
import joblib
import numpy as np
import pandas as pd

# Muat model KNN dan scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Muat dataset warna dari CSV
def load_color_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Fungsi untuk mendeteksi warna dan menghitung confidence score
def detect_color(pixel, color_data, max_confidence=1.90):
    pixel_scaled = scaler.transform([pixel])
    distances, indices = knn.kneighbors(pixel_scaled, n_neighbors=1)
    color_pred = knn.predict(pixel_scaled)[0]
    
    # Cek apakah warna ada dalam dataset
    if color_pred in color_data['color_name'].values:
        # Hitung confidence score berdasarkan jarak terdekat
        distance = distances[0][0]
        # Gunakan fungsi sigmoid untuk mengubah jarak menjadi skor antara 0 dan 1
        sigmoid_score = 1 / (1 + np.exp(distance))
        # Scaling confidence score agar mencapai nilai maksimal yang diinginkan
        confidence = max_confidence * sigmoid_score
        return color_pred, confidence
    else:
        return "Unknown Color", 0.0

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Muat dataset warna
color_data = load_color_dataset('colors.csv')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height // 2, width // 2]
    
    # Deteksi warna dan hitung confidence score
    color_name, confidence = detect_color(pixel_center, color_data)
    
    # Format confidence score dengan dua desimal
    confidence_score = f"{confidence:.2f}"
    
    # Tampilkan warna dan confidence score pada frame
    label = f"{color_name}: {confidence_score}"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
