import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from unet_model import UNet

# === 1. Dataset giả lập (nếu bạn chưa có ảnh thật) ===
class RandomShapesDataset(Dataset):
    def __init__(self, num_samples=100, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Vẽ hình tròn ngẫu nhiên
        center = np.random.randint(20, self.img_size - 20, 2)
        radius = np.random.randint(10, 30)
        color = np.random.randint(100, 255, 3).tolist()

        cv2.circle(img, tuple(center), radius, color, -1)
        cv2.circle(mask, tuple(center), radius, 255, -1)

        img = self.transform(img)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return img, mask


# === 2. Khởi tạo mô hình và các thành phần train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Using device:", device)

model = UNet(in_channels=3, out_channels=1).to(device)

# Load lại model cũ nếu có
try:
    model.load_state_dict(torch.load("unet_trained.pth", map_location=device))
    print("✅ Đã load model cũ, tiếp tục training...")
except FileNotFoundError:
    print("⚠️ Không tìm thấy model cũ, training mới.")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = RandomShapesDataset(num_samples=100, img_size=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# === 3. Vòng lặp huấn luyện ===
num_epochs = 25
losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)
        loss = criterion(torch.sigmoid(outputs), masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"📘 Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Lưu model mỗi 5 epoch
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")
        print(f"💾 Đã lưu model tại epoch {epoch + 1}")

# === 4. Lưu mô hình cuối cùng ===
torch.save(model.state_dict(), "unet_trained.pth")
print("✅ Training complete and model saved as unet_trained.pth")

# === 5. Vẽ biểu đồ loss ===
plt.plot(losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Biểu đồ Loss trong quá trình huấn luyện U-Net")
plt.grid(True)
plt.show()
