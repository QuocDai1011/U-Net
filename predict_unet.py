import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from unet_model import UNet

# === 1. Load model đã train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("unet_trained.pth", map_location=device))
model.eval()

# === 2. Đọc ảnh và tiền xử lý ===
image_path = "data/images/mouth1.jpg"  # Thay đường dẫn ảnh của bạn
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    print("⚠️ Không tìm thấy ảnh, tạo ảnh giả lập để test.")
    img_bgr = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(img_bgr, (64, 64), 25, (200, 120, 0), -1)

# Chuyển sang RGB và resize về (128,128)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (128, 128))

# Chuẩn hóa giống lúc train (0-1)
img_norm = img_resized / 255.0
img_tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

# === 3. Dự đoán mask ===
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]

print("🔍 Giá trị min/max của mask_pred:", pred_mask.min(), pred_mask.max())

# Nếu max nhỏ hơn 0.5 → hạ ngưỡng để xem có vùng sáng không
# Tự chọn ngưỡng rõ hơn
threshold = 0.7  # hoặc thử 0.6 / 0.7 để xem vùng mặt được tách tốt nhất
pred_mask_bin = (pred_mask > threshold).astype(np.uint8) * 255

kernel = np.ones((3,3), np.uint8)
pred_mask_bin = cv2.morphologyEx(pred_mask_bin, cv2.MORPH_OPEN, kernel)


# === 4. Hiển thị kết quả ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Ảnh gốc")
plt.imshow(img_rgb)

plt.subplot(1, 3, 2)
plt.title("Mask dự đoán (xám)")
plt.imshow(pred_mask, cmap='gray')

plt.subplot(1, 3, 3)
plt.title(f"Mask nhị phân (threshold={threshold})")
plt.imshow(pred_mask_bin, cmap='gray')

plt.tight_layout()
plt.show()
