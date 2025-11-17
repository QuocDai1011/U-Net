import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet

# ======= 1. Cấu hình =======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_mouth_trained.pth"
IMAGE_PATH = "data/test_image/thumb.jpg"  # ảnh test
OUTPUT_MASK_PATH = "data/result-test-mouth/result_mask.png"
OUTPUT_OVERLAY_PATH = "data/result-test-mouth/result_overlay.png"

os.makedirs(os.path.dirname(OUTPUT_MASK_PATH), exist_ok=True)

# ======= 2. Load model đã train =======
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model loaded thành công trên thiết bị:", DEVICE)

# ======= 3. Đọc và tiền xử lý ảnh đầu vào =======
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Không tìm thấy ảnh test: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original_size = img.shape[:2]

# 🔧 Cải thiện ánh sáng bằng CLAHE (chống sáng gắt, tối vùng)
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)
lab = cv2.merge((l, a, b))
img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# 🔧 Giảm nhiễu mịn (không làm mờ chi tiết)
img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

img_resized = cv2.resize(img, (256, 256))

# ======= 4. Chuẩn bị tensor =======
input_tensor = torch.from_numpy(img_resized / 255.0).float()
input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# ======= 5. Dự đoán mask =======
with torch.no_grad():
    pred = model(input_tensor)
    mask_pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

print(f"🔍 Giá trị mask dự đoán: min={mask_pred.min():.4f}, max={mask_pred.max():.4f}")

# ======= 6. Hậu xử lý mask =======
threshold = 0.7  # bạn có thể thử tăng lên 0.7 nếu mask vẫn quá rộng
mask_binary = (mask_pred > threshold).astype(np.uint8) * 255

# 🔧 Làm mịn mask bằng morphological operations
kernel = np.ones((5, 5), np.uint8)
mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)  # bỏ nhiễu nhỏ
mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel) # làm kín vùng
mask_binary = cv2.GaussianBlur(mask_binary, (3, 3), 0)               # làm mịn biên

mask_resized = cv2.resize(mask_binary, (original_size[1], original_size[0]))

# ======= 7. Hiển thị kết quả =======
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Ảnh gốc (đã tiền xử lý)")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask dự đoán (sau hậu xử lý)")
plt.imshow(mask_resized, cmap="gray")
plt.axis("off")

# Overlay
overlay = img.copy()
overlay[mask_resized > 127] = [255, 0, 0]
overlay = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

plt.subplot(1, 3, 3)
plt.title("Overlay (mask trên ảnh)")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

# ======= 8. Lưu kết quả =======
cv2.imwrite(OUTPUT_MASK_PATH, mask_resized)
cv2.imwrite(OUTPUT_OVERLAY_PATH, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"💾 Mask lưu tại: {OUTPUT_MASK_PATH}")
print(f"💾 Overlay lưu tại: {OUTPUT_OVERLAY_PATH}")
