# 🦷 Mouth Segmentation using U-Net / U-Net++
Dự án sử dụng mô hình **U-Net** và **U-Net++** để phân đoạn khoang miệng từ ảnh nha khoa, tạo ra **mask nhị phân** để nhận diện vùng miệng.

---

## 📌 1. Giới thiệu
Dự án này tập trung vào bài toán **Image Segmentation** trong lĩnh vực nha khoa — tách khoang miệng bằng **U-Net / U-Net++**.

### Pipeline gồm 4 bước:
- Tiền xử lý ảnh  
- Huấn luyện mô hình  
- Dự đoán → tạo mask nhị phân  
- Hiển thị kết quả (Before → Mask → Overlay)

---

## 🛠️ 2. Công nghệ sử dụng

| Thành phần  | Phiên bản              |
| ----------- | ---------------------- |
| Python      | 3.9.24                 |
| PyTorch     | GPU (CUDA)             |
| CUDA        | Optional (khuyến nghị) |
| OpenCV      | Latest                 |
| Anaconda    | Tạo môi trường         |
| torchvision | Data transforms        |

---

## ⚙️ 3. Cài đặt môi trường

### 🔹 3.1 Tạo môi trường bằng Conda
conda create -n unet_env python=3.9.24
conda activate unet_env

### 🔹 3.2 Cài PyTorch + CUDA (khuyến nghị)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### 🔹 3.3 Cài các thư viện còn lại
pip install opencv-python matplotlib numpy tqdm

---

# 📁 4. Cấu trúc thư mục
<img width="578" height="893" alt="image" src="https://github.com/user-attachments/assets/1a66c095-3297-4560-8476-1d8381a3eb6d" />


### 🔸 Lưu ý:
Tên file trong images/ và masks/ phải trùng nhau.
Ví dụ:
images/tooth01.jpg  ↔  masks/tooth01.png

---

# 🧹 5. Tiền xử lý ảnh
Dự án sử dụng các kỹ thuật:
Resize 256×256
Chuyển RGB
CLAHE (tăng độ tương phản)
Giảm nhiễu Gaussian
Data Augmentation:
HorizontalFlip
RandomRotation
ColorJitter

---

# 🧠 6. Huấn luyện mô hình
Thông số	Giá trị
Epoch	150
Loss	BCE
Optimizer	Adam
Learning rate	1e-4
Batch size	2

---

# ▶️ 7. Chạy huấn luyện
python train_mouth_unet.py

---

# 🔍 8. Chạy dự đoán
python test_mouth_unet.py
