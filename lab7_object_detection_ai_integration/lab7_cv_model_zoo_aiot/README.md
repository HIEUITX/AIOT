# Lab 7 mở rộng: Computer Vision Model Zoo for AIoT

Bản mở rộng này phát triển sau Lab 7 cơ bản. Mục tiêu là cho sinh viên trải nghiệm nhiều nhóm mô hình thị giác máy tính khác nhau, không chỉ YOLO.

## Chạy nhanh

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux/WSL
source .venv/bin/activate
pip install -r requirements_core.txt
python run_model_zoo_demo.py
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở trình duyệt: http://127.0.0.1:8000/

## Cấu trúc file chính

- `app.py`: backend FastAPI, camera stream, upload ảnh, log, event.
- `vision_engines.py`: các engine thị giác: detection, tracking, pose, hand, face, OCR, segmentation, motion.
- `index.html`: dashboard chọn task và quan sát kết quả.
- `run_model_zoo_demo.py`: smoke test không cần camera thật.

## Lưu ý

Các engine có fallback để bài lab luôn chạy được. Nếu muốn chạy model thật, cài thêm thư viện trong `requirements_optional.txt` và đọc tài liệu trong thư mục `docs/`.
