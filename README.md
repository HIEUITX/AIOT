TRƯỜNG ĐẠi HỌC ĐẠI NAM

MÔN:Triển khai, phát triển ứng dụng AI và IoT

BÁO CÁO LAB 2
Chuẩn bị dữ liệu IoT cho AI
và Deploy Baseline Model

Sinh viên:Nguyễn Văn Hiếu
Lớp:CNTT 17-01
GVHD:# AIOT Ths. Nguyễn Văn Nhân

1 NỘI DUNG CHÍNH  
Lab 2 tập trung xây dựng pipeline AIoT cơ bản gồm:
chuẩn bị dữ liệu IoT, train model AI baseline và deploy model bằng FastAPI.
Bài lab sử dụng dataset UCI Occupancy Detection để dự đoán trạng thái phòng học có người hay không dựa trên dữ liệu cảm biến như nhiệt độ, độ ẩm, ánh sáng và CO2.
Dataset sử dụng: UCI Occupancy Detection Dataset.
Dữ liệu gồm các trường:
- Temperature
- Humidity
- Light
- CO2
- HumidityRatio
- Occupancy
| Cột         | Ý nghĩa            |
| ----------- | ------------------ |
| Temperature | Nhiệt độ           |
| Humidity    | Độ ẩm              |
| Light       | Ánh sáng           |
| CO2         | Nồng độ CO2        |
| Occupancy   | Có người hay không |
2 PHẦN XỬ LÍ DỮ LIỆU
Dữ liệu được xử lý bằng pandas:
- Xóa dữ liệu trùng lặp
- Xử lý missing value
- Chuẩn hóa timestamp
- Loại bỏ outlier
Sau khi clean, dữ liệu được lưu vào:
telemetry_clean.csv
3 FEATURE ENGINEERING
Hệ thống tạo thêm feature:
- hour
- dayofweek
Các feature này giúp model học theo thời gian hoạt động của phòng học.
4 TRAIN MODEL AI
Model Logistic Regression được sử dụng để dự đoán occupancy.
Dữ liệu được chia train/test theo thời gian với tỉ lệ 75/25 để tránh data leakage.
METRIC:
{
  "accuracy": 0.9943579766536965,
  "precision": 0.9750889679715302,
  "recall": 0.9990884229717412,
  "f1": 0.9869428185502026,
  "roc_auc": 0.9988282300727526,
  "confusion_matrix": [
    [
      4015,
      28
    ],
    [
      1,
      1096
    ]
  ]

}
Model sau khi train được lưu thành file .joblib.
FastAPI được sử dụng để deploy model thành API local với các endpoint:
- /health
- /model-info
- /predict
=> KẾT QUẢ
Pipeline AIoT đã hoạt động thành công:
- Clean dữ liệu IoT
- Train AI model
- Tạo anomaly detection
- Deploy API bằng FastAPI
- Predict dữ liệu mới qua endpoint /predict
