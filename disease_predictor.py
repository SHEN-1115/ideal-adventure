import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split


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
if st.button("預測"):
    if selected_symptoms:
        # 建立 one-hot 輸入向量
        input_vector = np.array([[symptom in selected_symptoms for symptom in filtered_df.columns[1:]]])
        prediction = model.predict(input_vector)
        disease = selected_diseases[np.argmax(prediction)]
        st.success(f"可能的疾病為：{disease}")
    else:
        st.warning("請選擇至少一個症狀！")
