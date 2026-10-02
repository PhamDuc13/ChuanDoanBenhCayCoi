import os
import shutil
import random

# Tự động nhận diện tên thư mục ảnh gốc của bạn
RAW_DIR = "PlantVillage_raw" if os.path.exists("PlantVillage_raw") else "color"
OUTPUT_DIR = "dataset_clean"
SPLIT_RATIO = 0.8
MAX_PER_CLASS = 250  # Mỗi lớp lấy 250 ảnh để máy train nhanh và nhẹ

def run_prepare():
    if not os.path.exists(RAW_DIR):
        print(f"LỖI: Không tìm thấy thư mục ảnh '{RAW_DIR}'. Hãy kiểm tra lại tên thư mục!")
        return

    print("[1/3] Đang quét các danh mục cây trồng và mầm bệnh...")
    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    print(f"-> Tìm thấy {len(classes)} danh mục bệnh.")

    # Xóa thư mục cũ nếu có để làm sạch
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for cls in classes:
        src_folder = os.path.join(RAW_DIR, cls)
        imgs = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        random.seed(42)
        random.shuffle(imgs)
        imgs = imgs[:MAX_PER_CLASS]

        split_idx = int(len(imgs) * SPLIT_RATIO)
        train_imgs = imgs[:split_idx]
        val_imgs = imgs[split_idx:]

        for subset, subset_imgs in [("train", train_imgs), ("val", val_imgs)]:
            dst_dir = os.path.join(OUTPUT_DIR, subset, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img_path in subset_imgs:
                shutil.copyfile(img_path, os.path.join(dst_dir, os.path.basename(img_path)))

    print(f"\n[XONG BƯỚC 1] Dữ liệu đã sẵn sàng trong thư mục '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    run_prepare()
