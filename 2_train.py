import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 5
DATASET_DIR = "dataset_clean"

def run_train():
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(train_dir):
        print(f"LỖI: Chưa có thư mục '{DATASET_DIR}'. Hãy chạy file '1_prepare.py' trước!")
        return

    # Data Augmentation giúp mô hình nhận diện tốt hơn khi đưa ảnh bên ngoài vào
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )

    # Lưu file danh sách nhãn lớp để file app đọc
    with open("class_indices.json", "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=4)

    num_classes = len(train_gen.class_indices)
    print(f"-> Tổng số bệnh và cây mô hình sẽ nhận diện: {num_classes}")

    # Xây dựng mạng MobileNetV2
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n[2/3] Bắt đầu huấn luyện mô hình (5 Epochs)...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    # Lưu mô hình hoàn thiện
    model.save("final_plant_model.h5")
    print("\n[XONG BƯỚC 2] Đã lưu mô hình vào 'final_plant_model.h5' và 'class_indices.json'!")

if __name__ == "__main__":
    run_train()
