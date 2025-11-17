import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unet_model import UNet
import numpy as np

# === Hàm tiền xử lý ảnh ===
def preprocess_image(img):
    # 1️⃣ Chuyển sang không gian màu LAB để cân bằng sáng
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Cân bằng sáng bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 2️⃣ Giảm nhiễu bằng bộ lọc Bilateral (giữ biên tốt hơn Gaussian)
    img_denoised = cv2.bilateralFilter(img_clahe, d=7, sigmaColor=75, sigmaSpace=75)

    # 3️⃣ Chuẩn hóa giá trị pixel [0, 1]
    img_norm = img_denoised / 255.0
    return (img_norm * 255)

# === Dataset đọc ảnh thật ===
class MouthDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Lỗi đọc ảnh hoặc mask: {img_name}")

        # ✅ Tiền xử lý
        img = preprocess_image(img)

        # Resize đồng nhất
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # Chuyển sang Tensor
        img = self.transform(img)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return img, mask


# === Cấu hình model và train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⚙️ Using device:", device)

dataset = MouthDataset("data/images", "data/masks", img_size=256)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Huấn luyện ===
for epoch in range(150):
    model.train()
    total_loss = 0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/150] - Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "unet_mouth_trained_preprocessed.pth")
print("✅ Đã huấn luyện xong và lưu model: unet_mouth_trained_preprocessed.pth")
