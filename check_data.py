import cv2, glob, os

image_paths = sorted(glob.glob("data/images/*"))
mask_paths = sorted(glob.glob("data/masks/*"))

print("Tổng số ảnh:", len(image_paths))
print("Tổng số mask:", len(mask_paths))

for i in range(min(3, len(image_paths))):
    img = cv2.imread(image_paths[i])
    mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
    print(f"\nẢnh {i+1}: {os.path.basename(image_paths[i])}")
    print("  ➤ img shape:", img.shape, " | min/max:", img.min(), img.max())
    print("  ➤ mask shape:", mask.shape, " | min/max:", mask.min(), mask.max())

    cv2.imshow("Ảnh", img)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
cv2.destroyAllWindows()
