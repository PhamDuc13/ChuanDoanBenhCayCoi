import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

MODEL_PATH = "final_plant_model.h5"
LABELS_PATH = "class_indices.json"
INPUT_SHAPE = (224, 224)

# Bảng dịch tên bệnh sang Tiếng Việt và biện pháp xử lý
VIETNAMESE_DISEASES = {
    "Apple___Apple_scab": ("Cây Táo: Bệnh vảy táo (Apple Scab)", "Phun thuốc gốc đồng hoặc Mancozeb vào đầu mùa mưa."),
    "Apple___Black_rot": ("Cây Táo: Bệnh thối đen (Black Rot)", "Cắt tỉa cành chết, thu gom tiêu hủy lá rụng."),
    "Apple___Cedar_apple_rust": ("Cây Táo: Bệnh rỉ sắt Cedar", "Dùng thuốc diệt nấm đặc trị rỉ sắt (Myclobutanil)."),
    "Apple___healthy": ("Cây Táo: Khỏe mạnh", "Tán lá bình thường, duy trì dinh dưỡng định kỳ."),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Cây Ngô: Bệnh đốm xám Cercospora", "Luân canh cây trồng, dọn tàn dư thực vật sau thu hoạch."),
    "Corn_(maize)___Common_rust_": ("Cây Ngô: Bệnh rỉ sắt thông thường", "Sử dụng giống kháng bệnh, phun thuốc diệt nấm phổ rộng."),
    "Corn_(maize)___Northern_Leaf_Blight": ("Cây Ngô: Bệnh cháy lá phương Bắc", "Tỉa bớt lá già sát gốc tạo độ thông thoáng."),
    "Corn_(maize)___healthy": ("Cây Ngô: Khỏe mạnh", "Cây sinh trưởng tốt, không có dấu hiệu bệnh."),
    "Grape___Black_rot": ("Cây Nho: Bệnh thối đen", "Phun thuốc phòng trừ nấm trước khi nảy chồi và sau mưa."),
    "Grape___Esca_(Black_Measles)": ("Cây Nho: Bệnh sởi đen Esca", "Khử trùng kỹ kéo cắt cành, cách ly cành bị nhiễm nặng."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Cây Nho: Bệnh cháy lá Isariopsis", "Phun thuốc gốc đồng định kỳ 10-14 ngày/lần."),
    "Grape___healthy": ("Cây Nho: Khỏe mạnh", "Cây xanh tốt, quang hợp bình thường."),
    "Orange___Haunglongbing_(Citrus_greening)": ("Cây Cam: Bệnh vàng lá gân xanh (Greening)", "Tiêu hủy cây nhiễm bệnh nặng, phun thuốc diệt rầy chổng cánh."),
    "Peach___Bacterial_spot": ("Cây Đào: Bệnh đốm lá vi khuẩn", "Phun chế phẩm vi khuẩn gốc đồng trước mùa mưa."),
    "Peach___healthy": ("Cây Đào: Khỏe mạnh", "Lá xanh đều, không phát hiện sâu nấm."),
    "Pepper,_bell___Bacterial_spot": ("Ớt chuông: Bệnh đốm vi khuẩn", "Tránh tưới phun sương lên mặt lá, phun hoạt chất gốc đồng hoặc Kasumin."),
    "Pepper,_bell___healthy": ("Ớt chuông: Khỏe mạnh", "Cây phát triển bình thường, duy trì tưới tiêu."),
    "Potato___Early_blight": ("Cây Khoai tây: Bệnh cháy lá sớm", "Phun thuốc diệt nấm chứa hoạt chất Mancozeb hoặc Chlorothalonil."),
    "Potato___Late_blight": ("Cây Khoai tây: Bệnh mốc sương mai", "Cực kỳ nguy hiểm: Cách ly ngay và phun thuốc đặc trị Ridomil Gold."),
    "Potato___healthy": ("Cây Khoai tây: Khỏe mạnh", "Lá căng đều, sinh trưởng tốt."),
    "Tomato___Bacterial_spot": ("Cây Cà chua: Bệnh đốm lá vi khuẩn", "Phun dung dịch Booc-đô hoặc thuốc gốc đồng."),
    "Tomato___Early_blight": ("Cây Cà chua: Bệnh cháy lá sớm", "Cắt tỉa lá già sát gốc, phun Daconil hoặc Mancozeb."),
    "Tomato___Late_blight": ("Cây Cà chua: Bệnh mốc sương mai", "Bệnh gây úng thâm lá nhanh, phun khẩn cấp Ridomil Gold."),
    "Tomato___Leaf_Mold": ("Cây Cà chua: Bệnh mốc lá", "Giảm độ ẩm vườn ươm, tăng cường thông gió."),
    "Tomato___Septoria_leaf_spot": ("Cây Cà chua: Bệnh đốm lá Septoria", "Dọn dẹp lá rụng, phun thuốc trừ nấm gốc đồng."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Cây Cà chua: Nhiễm nhện đỏ gây hại", "Tăng ẩm độ không khí, phun dầu khoáng hoặc thuốc diệt nhện."),
    "Tomato___Target_Spot": ("Cây Cà chua: Bệnh đốm mục tiêu Target Spot", "Phun luân phiên các hoạt chất diệt nấm phổ rộng."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Cây Cà chua: Bệnh virus xoăn vàng lá", "Nhổ bỏ cây bệnh để tránh lây nhiễm, phun bẫy dính diệt bọ phấn trắng."),
    "Tomato___Tomato_mosaic_virus": ("Cây Cà chua: Bệnh virus khảm lá Mosaic", "Khử trùng đất và hạt giống, nhổ bỏ cây bệnh."),
    "Tomato___healthy": ("Cây Cà chua: Khỏe mạnh", "Diệp lục phân bố hoàn hảo, cây rất khỏe mạnh.")
}

classifier_model = None
labels_map = {}

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    try:
        classifier_model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            class_dict = json.load(f)
            labels_map = {v: k for k, v in class_dict.items()}
    except Exception as e:
        print("Lỗi nạp mô hình:", e)

class CleanPlantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AgriScan AI - Chẩn đoán tên bệnh lá cây chi tiết")
        self.root.geometry("1020x680")
        self.root.configure(bg="#F4F7F6")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Green.Horizontal.TProgressbar", troughcolor="#E0E0E0", background="#27AE60", thickness=14)
        self.style.configure("Red.Horizontal.TProgressbar", troughcolor="#E0E0E0", background="#E74C3C", thickness=14)
        self.style.configure("Orange.Horizontal.TProgressbar", troughcolor="#E0E0E0", background="#E67E22", thickness=14)

        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1B5E20", height=75)
        header.pack(fill="x")
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg="#1B5E20")
        title_box.pack(side="left", padx=25, pady=12)
        tk.Label(title_box, text="🌿 CHẨN ĐOÁN CHI TIẾT TÊN BỆNH CÂY TRỒNG (AI)", font=("Segoe UI", 16, "bold"), bg="#1B5E20", fg="white").pack(anchor="w")
        tk.Label(title_box, text="Phân tích triệu chứng và tên bệnh chuẩn xác ứng dụng MobileNetV2", font=("Segoe UI", 9), bg="#1B5E20", fg="#C8E6C9").pack(anchor="w")

        content = tk.Frame(self.root, bg="#F4F7F6")
        content.pack(fill="both", expand=True, padx=25, pady=20)

        # Cột trái
        left = tk.Frame(content, bg="white", bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))
        tk.Label(left, text="ẢNH MẪU ĐANG KIỂM TRA", font=("Segoe UI", 11, "bold"), bg="white").pack(anchor="w", padx=20, pady=(15, 10))

        self.preview = tk.Label(left, text="Chưa nạp ảnh\n\nNhấn nút bên dưới để chọn ảnh", bg="#ECEFF1", fg="#78909C", font=("Segoe UI", 10))
        self.preview.pack(fill="both", expand=True, padx=20, pady=5)
        self.lbl_info = tk.Label(left, text="Định dạng: ---", font=("Segoe UI", 8), bg="white", fg="#757575")
        self.lbl_info.pack(pady=5)
        tk.Button(left, text="📂 TẢI ẢNH LÁ CÂY LÊN", font=("Segoe UI", 10, "bold"), bg="#2E7D32", fg="white", relief="flat", cursor="hand2", pady=10, command=self.load_image).pack(fill="x", padx=20, pady=(5, 20))

        # Cột phải
        right = tk.Frame(content, bg="white", bd=1, relief="solid")
        right.pack(side="right", fill="both", expand=True, padx=(15, 0))
        tk.Label(right, text="KẾT QUẢ CHẨN ĐOÁN & TÊN BỆNH CỤ THỂ", font=("Segoe UI", 11, "bold"), bg="white").pack(anchor="w", padx=20, pady=(15, 10))

        self.card = tk.Frame(right, bg="#E8F5E9", padx=15, pady=12)
        self.card.pack(fill="x", padx=20, pady=10)

        self.lbl_title = tk.Label(self.card, text="ĐANG CHỜ ẢNH ĐẦU VÀO", font=("Segoe UI", 13, "bold"), bg="#E8F5E9", fg="#27AE60")
        self.lbl_title.pack(anchor="w")
        self.lbl_sub = tk.Label(self.card, text="Hệ thống hỗ trợ chẩn đoán chính xác tên từng loại bệnh.", font=("Segoe UI", 9), bg="#E8F5E9", fg="#455A64")
        self.lbl_sub.pack(anchor="w", pady=(3, 0))

        bar_box = tk.Frame(right, bg="white")
        bar_box.pack(fill="x", padx=20, pady=(10, 5))
        tk.Label(bar_box, text="Độ tin cậy của mô hình:", font=("Segoe UI", 9, "bold"), bg="white").pack(side="left")
        self.lbl_conf = tk.Label(bar_box, text="0.0%", font=("Segoe UI", 10, "bold"), bg="white", fg="#27AE60")
        self.lbl_conf.pack(side="right")

        self.progress = ttk.Progressbar(right, style="Green.Horizontal.TProgressbar", length=100)
        self.progress.pack(fill="x", padx=20, pady=(0, 15))

        rec_box = tk.LabelFrame(right, text=" Hướng dẫn xử lý nông học & Đơn thuốc khuyến nghị ", font=("Segoe UI", 9, "bold"), bg="white", padx=12, pady=10)
        rec_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.txt_rec = tk.Text(rec_box, font=("Segoe UI", 9), bg="#FAFAFA", wrap="word", bd=0, padx=8, pady=8)
        self.txt_rec.pack(fill="both", expand=True)
        self.txt_rec.insert("1.0", "Tên bệnh học chi tiết và cách điều trị sẽ hiển thị tại đây.")
        self.txt_rec.config(state="disabled")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.webp")])
        if not file_path:
            return

        img = Image.open(file_path)
        self.lbl_info.config(text=f"Tệp: {os.path.basename(file_path)} | Kích thước: {img.size[0]}x{img.size[1]} px")

        preview = img.copy()
        preview.thumbnail((360, 360))
        tk_img = ImageTk.PhotoImage(preview)
        self.preview.configure(image=tk_img, text="")
        self.preview.image = tk_img

        self.predict(img)

    def predict(self, pil_img):
        if classifier_model is None:
            messagebox.showerror("Lỗi", "Chưa tìm thấy final_plant_model.h5! Hãy chạy 2_train.py trước!")
            return

        img_prep = pil_img.convert("RGB").resize(INPUT_SHAPE)
        tensor = np.array(img_prep, dtype=np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        preds = classifier_model.predict(tensor)[0]
        class_idx = int(np.argmax(preds))
        confidence = preds[class_idx] * 100.0
        raw_name = labels_map.get(class_idx, "Unknown")

        # NGƯỠNG AN TOÀN CHO LÁ NGOÀI DANH MỤC
        if confidence < 50.0:
            title = "❓ KHÔNG XÁC ĐỊNH ĐƯỢC BỆNH (ĐỐI TƯỢNG LẠ)"
            sub = "Độ tin cậy thấp: Chiếc lá này có thể không thuộc danh mục bệnh cây trồng đã học."
            color = "#E67E22"
            bg = "#FFF3E0"
            style_name = "Orange.Horizontal.TProgressbar"
            advice = (
                "Khuyến nghị chuyên môn:\n"
                "• Hình ảnh lá cây không thuộc cơ sở dữ liệu huấn luyện chuẩn (Apple, Corn, Grape, Tomato, Potato, Pepper...).\n"
                "• Vui lòng chụp rõ một phiến lá dưới điều kiện đủ sáng hoặc bổ sung dữ liệu cho giống cây này."
            )
        elif "healthy" in raw_name.lower():
            info = VIETNAMESE_DISEASES.get(raw_name, (raw_name, "Cây đang sinh trưởng tốt."))
            title = f"✔ {info[0].upper()}"
            sub = "Tình trạng: Tán lá bình thường, không có dấu hiệu nấm hoặc vi khuẩn."
            color = "#27AE60"
            bg = "#E8F5E9"
            style_name = "Green.Horizontal.TProgressbar"
            advice = f"Tình trạng: Sức khỏe tốt.\n\nBiện pháp: {info[1]}"
        else:
            info = VIETNAMESE_DISEASES.get(raw_name, (raw_name, "Cần cách ly và phun thuốc bảo vệ thực vật."))
            title = f"⚠ {info[0].upper()}"
            sub = f"Mã bệnh gốc: {raw_name}"
            color = "#C0392B"
            bg = "#FFEBEE"
            style_name = "Red.Horizontal.TProgressbar"
            advice = f"1. Tên bệnh học: {info[0]}\n\n2. Hướng dẫn điều trị & Khuyến cáo:\n• {info[1]}"

        self.card.configure(bg=bg)
        self.lbl_title.configure(text=title, fg=color, bg=bg)
        self.lbl_sub.configure(text=sub, bg=bg)
        self.lbl_conf.configure(text=f"{confidence:.2f}%", fg=color)
        self.progress.configure(style=style_name)
        self.progress["value"] = confidence

        self.txt_rec.config(state="normal")
        self.txt_rec.delete("1.0", "end")
        self.txt_rec.insert("1.0", advice)
        self.txt_rec.config(state="disabled")

if __name__ == "__main__":
    w = tk.Tk()
    app = CleanPlantApp(w)
    w.mainloop()
