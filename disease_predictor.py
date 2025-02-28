import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split

# 讀取 CSV
file_path = "https://raw.githubusercontent.com/SHEN-1115/ideal-adventure/main/disease.csv"
df = pd.read_csv(file_path)
df["Disease"] = df["Disease"].str.strip()

# 清理症狀資料
def clean_symptoms(df):
    symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
    df["symptoms"] = df[symptom_cols].apply(lambda row: [s for s in row if pd.notna(s)], axis=1)
    return df.drop(columns=symptom_cols)

def create_symptom_table(df):
    unique_symptoms = set(s for symptoms in df["symptoms"] for s in symptoms)
    table = pd.DataFrame([{symptom: (symptom in row["symptoms"]) for symptom in unique_symptoms} for _, row in df.iterrows()])
    table.insert(0, "Disease", df["Disease"].values)
    return table

cleaned_df = clean_symptoms(df)
symptom_table = create_symptom_table(cleaned_df)

# 篩選特定疾病
selected_diseases = [
    "Allergy", "Common Cold", "Acne", "Hypertension", "GERD",
    "Fungal infection", "Urinary tract infection", "Psoriasis", "Migraine",
    "Osteoarthristis", "Cervical spondylosis", "Peptic ulcer diseae",
    "Bronchial Asthma", "Varicose veins"
]
filtered_df = symptom_table[symptom_table["Disease"].isin(selected_diseases)]
filtered_df = filtered_df.sort_values(by="Disease").reset_index(drop=True)

# 轉換為 One-Hot Encoding
Y = pd.get_dummies(filtered_df['Disease'])
X = filtered_df[[col for col in filtered_df.columns if col != 'Disease']].astype(int)

# 分割訓練與測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 建立神經網路模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(Y_train.shape[1], activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

# 儲存模型
model.save("disease_model.keras")  # 儲存成 .keras 格式

# Streamlit 頁面
st.title("疾病診斷模型")

# 使用 @st.cache_resource 裝飾器來緩存模型加載
@st.cache_resource
def load_model():
    # 確保模型只會加載一次
    return tf.keras.models.load_model("disease_model.keras")

# 載入模型（這一步只會執行一次，緩存起來）
model = load_model()

# 建立一個症狀選擇框
selected_symptoms = st.multiselect("選擇症狀", list(filtered_df.columns[1:]))
if st.button("預測"):
    if selected_symptoms:
        # 建立 one-hot 輸入向量
        input_vector = np.array([[symptom in selected_symptoms for symptom in filtered_df.columns
