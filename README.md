# 🧠 Bank Customer Churn Prediction

Dự án này sử dụng mô hình Deep Learning (MLP - Multilayer Perceptron) để dự đoán khả năng khách hàng rời bỏ ngân hàng (churn). Bao gồm:
- REST API với **FastAPI**
- Giao diện người dùng với **Streamlit**
- Mô hình đã huấn luyện (`model.h5`)
- Scaler đã huấn luyện (`scaler.pkl`)

---

## 📁 Cấu trúc thư mục

```
bank_churn_app/
├── main.py              # REST API bằng FastAPI
├── app.py               # Giao diện Streamlit
├── model.h5             # Mô hình đã huấn luyện
├── scaler.pkl           # Scaler đã lưu
├── requirements.txt     # Danh sách thư viện cần cài
```

---

## 🚀 Cài đặt

### 🔧 1. Tạo môi trường ảo (tuỳ chọn)
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 📦 2. Cài thư viện
```bash
pip install -r requirements.txt
```

---

## ⚙️ Chạy REST API với FastAPI

### ▶️ 1. Chạy API
```bash
uvicorn main:app --reload
```

### 🌐 2. Truy cập Swagger UI:
Mở trình duyệt và truy cập:

```
http://127.0.0.1:8000/docs
```

Tại đây bạn có thể thử POST request với dữ liệu mẫu:

```json
{
  "CreditScore": 650,
  "Gender": 1,
  "Age": 35,
  "Tenure": 3,
  "Balance": 10000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```

---

## 🖥️ Chạy giao diện người dùng với Streamlit

```bash
streamlit run app.py
```

Giao diện sẽ xuất hiện ở địa chỉ:

```
http://localhost:8501
```

Bạn có thể nhập thông tin khách hàng và nhận dự đoán trực tiếp.

---

## 📌 Yêu cầu thư viện (nếu không dùng requirements.txt)

```txt
tensorflow
fastapi
uvicorn
streamlit
scikit-learn
joblib
```

Cài đặt thủ công:
```bash
pip install tensorflow fastapi uvicorn streamlit scikit-learn joblib
```

---

## 📬 Liên hệ

**Người phát triển:** *Tên bạn*  
**Email:** *email@example.com*
