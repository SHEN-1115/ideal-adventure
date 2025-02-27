# -*- coding: utf-8 -*-
"""Disease_Symptom_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://raw.githubusercontent.com/SHEN-1115/ideal-adventure/refs/heads/main/disease.csv

# 練習一：下載資料集、讀取資料集
"""



"""## 下載資料集

"""

import pandas as pd
file_path = "https://raw.githubusercontent.com/SHEN-1115/ideal-adventure/refs/heads/main/disease.csv"
df = pd.read_csv(file_path)
df.describe()

"""## 讀取資料集"""

from sklearn import tree
import numpy as np
import pandas as pd
df = pd.read_csv("diseases.csv")
print(df.shape)
df

"""# 練習二：清理資料、資料轉換、補值"""

def clean_symptoms(df):
    # 找出所有症狀欄位
    symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]

    # 合併症狀並去除 NaN
    df["symptoms"] = df[symptom_cols].apply(lambda row: [s for s in row if pd.notna(s)], axis=1)

    # 刪除原始症狀欄位
    df = df.drop(columns=symptom_cols)

    return df

# 1️⃣ 清理症狀資料
df["Disease"] = df["Disease"].str.strip()
def clean_symptoms(df):
    symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
    df["symptoms"] = df[symptom_cols].apply(lambda row: [s for s in row if pd.notna(s)], axis=1)
    df = df.drop(columns=symptom_cols)  # 刪除原始症狀欄位
    return df

# 2️⃣ 建立 True/False 表格
def create_symptom_table(df):
    # 取得所有獨特症狀
    unique_symptoms = set(s for symptoms in df["symptoms"] for s in symptoms)

    # 建立 one-hot encoding 表格
    table = pd.DataFrame([
        {symptom: (symptom in row["symptoms"]) for symptom in unique_symptoms}
        for _, row in df.iterrows()
    ])

    # 加入疾病名稱
    table.insert(0, "Disease", df["Disease"].values)

    return table

cleaned_df = clean_symptoms(df)
symptom_table = create_symptom_table(cleaned_df)

pd.set_option("display.max_columns", None)  # 允許顯示所有欄位
pd.set_option("display.width", 200)  # 調整顯示寬度
print(symptom_table)

# 選擇要保留的疾病列表
selected_diseases = [
    "Allergy", "Common Cold", "Acne", "Hypertension", "GERD",
    "Fungal infection", "Urinary tract infection", "Psoriasis", "Migraine",
    "Osteoarthristis", "Cervical spondylosis", "Peptic ulcer diseae",
    "Bronchial Asthma", "Varicose veins"
]

# 篩選 dataframe，只保留這些疾病
filtered_df = symptom_table[symptom_table["Disease"].isin(selected_diseases)]

#依照 Disease 進行a-z排序
filtered_df = filtered_df.sort_values(by="Disease").reset_index(drop=True)

# 顯示篩選後的資料
print(filtered_df)

"""# 練習四：訓練模型"""

print(filtered_df['Disease'].unique())
Y = pd.get_dummies(filtered_df['Disease'])
print(Y.head())  # 檢查 Y 的前幾行

from sklearn.model_selection import train_test_split

# 將 'Disease' 欄位轉換為 One-Hot 編碼
Y = pd.get_dummies(filtered_df['Disease'])

# 選擇症狀欄位 (不包括 'Disease')
X = filtered_df[[col for col in filtered_df.columns if col != 'Disease']].astype(int)

# 使用 train_test_split 將資料分為訓練組和測試組
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 確認訓練組和測試組的形狀
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# 建立神經網路模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # 第一層
    keras.layers.Dense(64, activation='relu'),  # 隱藏層
    keras.layers.Dense(Y_train.shape[1], activation='softmax')  # 輸出層 (One-Hot 編碼的欄位數)
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

# 在測試資料上評估模型準確性
test_loss, test_accuracy = model.evaluate(X_test, Y_test)

# 顯示測試結果
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 在訓練資料上評估模型準確性
train_loss, train_accuracy = model.evaluate(X_train, Y_train)

# 顯示訓練結果
print(f"Train Loss: {train_loss}")
print(f"Train Accuracy: {train_accuracy}")
