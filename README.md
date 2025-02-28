# Trích Xuất Thông Tin Căn Cước Công Dân

Đây là một dự án sử dụng Gradio để trích xuất thông tin từ ảnh căn cước công dân. Phần mềm sử dụng mô hình YOLO để phát hiện các vùng chứa thông tin, cùng với các công cụ OCR (vietOCR và PaddleOCR) để nhận diện chữ viết.

## Yêu Cầu

Để chạy được chương trình, bạn cần tải và cài đặt các file sau:

1. **File trọng số của YOLO và ảnh chuẩn**
   - **Tên file:** `info_detection.pt` và `anchor.jpg`
   - **Link tải:** [Ở đây](https://drive.google.com/drive/folders/1IlJGSQp7N5JLvW9lUa8cSQeIIBvHCCZo?usp=sharing)

> **Lưu ý:** Sau khi tải về, hãy đặt các file này vào cùng thư mục với file mã nguồn chính của dự án để chương trình có thể truy cập đúng.

## Cài Đặt Các Thư Viện Phụ Thuộc

Cài đặt các thư viện cần thiết bằng pip:

```bash
pip install gradio opencv-python numpy Pillow ultralytics vietocr paddleocr
