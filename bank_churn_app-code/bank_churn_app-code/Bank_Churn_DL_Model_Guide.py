# 🧠 Bank Customer Churn Prediction - Deep Learning Model

# ✅ 1. Cài đặt thư viện
# pip install pandas numpy scikit-learn tensorflow matplotlib seaborn

# 📁 2. Đọc và xử lý dữ liệu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("churn_data.csv")

# Mã hóa biến phân loại
label_enc = LabelEncoder()
df['Gender'] = label_enc.fit_transform(df['Gender'])  # ví dụ: Male -> 1, Female -> 0

# Chọn đặc trưng và nhãn
X = df.drop('Exited', axis=1)
y = df['Exited']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 🏗️ 3. Xây dựng mô hình Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

# 🏃 4. Huấn luyện mô hình
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# 📈 5. Đánh giá mô hình
from sklearn.metrics import classification_report, roc_auc_score

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))

# 📊 6. Vẽ biểu đồ lịch sử huấn luyện
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 💾 7. Lưu mô hình
model.save("model.h5")
