import cv2
import joblib
import numpy as np
import csv  # Impor modul csv

# Muat model KNN dan scaler
try:
    knn = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan. {e}")
    exit()
except Exception as e:
    print(f"Error: Terjadi kesalahan saat memuat model. {e}")
    exit()

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

frame_number = 0

# Buka file CSV untuk menulis
try:
    with open('color_predictions.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Color'])  # Tulis header

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera.")
                break
            
            frame_number += 1
            
            # Ambil pixel tengah gambar
            height, width, _ = frame.shape
            pixel_center = frame[height//2, width//2]
            
            # Normalisasi pixel sebelum prediksi
            try:
                pixel_center_scaled = scaler.transform([pixel_center])
            except Exception as e:
                print(f"Error: Terjadi kesalahan saat normalisasi pixel. {e}")
                continue
            
            # Prediksi warna
            try:
                color_pred = knn.predict(pixel_center_scaled)[0]
            except Exception as e:
                print(f"Error: Terjadi kesalahan saat melakukan prediksi. {e}")
                continue
            
            # Tampilkan warna pada frame
            cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Cetak hasil prediksi ke konsol
            print(f"Frame: {frame_number}, Color: {color_pred}")
            
            # Tulis hasil prediksi ke file CSV
            try:
                csvwriter.writerow([frame_number, color_pred])
            except Exception as e:
                print(f"Error: Terjadi kesalahan saat menulis ke file CSV. {e}")
                continue
            
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"Error: Terjadi kesalahan saat membuka atau menulis file CSV. {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
