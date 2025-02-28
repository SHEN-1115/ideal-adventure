import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split

################################################################################################

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

####################################################################################

# Streamlit 頁面
st.title("疾病診斷模型")

# 使用 @st.cache_resource 裝飾器來緩存模型加載
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("disease_model.keras")

# 載入模型
model = load_model()

# 建立一個症狀選擇框
selected_symptoms = st.multiselect("選擇症狀", list(filtered_df.columns[1:]))

# 定義 input_vector 預設值為 None
input_vector = None

if st.button("預測"):
    if selected_symptoms:
        # 建立 one-hot 輸入向量
        input_vector = np.array([[symptom in selected_symptoms for symptom in filtered_df.columns[1:]]])
        
        # 將布林值轉換為浮點數 (0 和 1)，並確保類型為 float32
        input_vector = input_vector.astype(np.float32)

        # 顯示輸入向量的形狀
        st.write(f"Input vector shape: {input_vector.shape}")
        st.write(f"Input vector: {input_vector}")

        # 預測並顯示結果
        prediction = model.predict(input_vector)
        disease = selected_diseases[np.argmax(prediction)]
        st.success(f"可能的疾病為：{disease}")
    else:
        st.warning("請選擇至少一個症狀！")
