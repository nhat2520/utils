import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_image_mask_pairs(images, masks, num_pairs=2):
    num_images = len(images)
    num_rows = num_images // num_pairs + int(num_images % num_pairs != 0)
    
    plt.figure(figsize=(8 * num_pairs, 5 * num_rows))
    idx = 0
    for row in range(num_rows):
        for pair in range(num_pairs):
            if idx >= num_images:
                break
            # Hiển thị ảnh gốc
            plt.subplot(num_rows, num_pairs * 2, row * num_pairs * 2 + pair * 2 + 1)
            plt.imshow(images[idx], cmap='gray')
            plt.title(f"Image {idx+1}")
            plt.axis('off')
            # Hiển thị mask
            plt.subplot(num_rows, num_pairs * 2, row * num_pairs * 2 + pair * 2 + 2)
            plt.imshow(masks[idx], cmap='gray')
            plt.title(f"Mask {idx+1}")
            plt.axis('off')
            idx += 1
            
    plt.tight_layout()
    plt.show()


def apply_mask_with_outline(image: np.ndarray,
                            mask: np.ndarray,
                            mask_color: tuple = (0, 255, 0),
                            alpha: float = 0.4,
                            outline_color: tuple = (0, 255, 0),
                            thickness: int = 2) -> np.ndarray:
    """
    Áp mask lên ảnh với fill và contour.

    - image: ảnh BGR (H×W×3)
    - mask: ảnh nhị phân (H'×W') hoặc float [0,255]
    - mask_color: (B,G,R)
    - alpha: độ trong suốt (0–1)
    - outline_color: (B,G,R)
    - thickness: độ dày contour
    """
    h, w = image.shape[:2]
    # nếu mask khác kích thước, resize về H×W
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # nhị phân
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # tạo ảnh màu fill
    colored_mask = np.zeros_like(image, dtype=image.dtype)
    colored_mask[:] = mask_color
    # chỉ giữ vùng mask
    colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=bin_mask)
    # overlay: blend image + colored_mask
    overlay = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

    # tìm contour & vẽ lên overlay
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, outline_color, thickness)

    return overlay
    