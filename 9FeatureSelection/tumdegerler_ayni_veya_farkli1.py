# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:52:22 2024

@author: akara
"""

import pandas as pd

# Örnek veri seti
data = {
    "A": ["a", "a", "a", "a"],  # Tüm değerleri aynı
    "B": ["a", "b", "c", "d"],  # Değerler farklı
    "C": ["x", "x", "x", "x"],   # Tüm değerleri aynı
    "D":[2,4,2,6]
}

df = pd.DataFrame(data)

print("Orijinal Veri Seti:")
print(df)

# Tüm değerleri aynı olan kategorik sütunları tespit etme
same_columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
# Tüm değerleri farklı olan kategorik sütunları tespit etme
diferent_columns_to_drop = [col for col in df.columns if df[col].nunique() == df.shape[0]]

drop_columns=same_columns_to_drop+diferent_columns_to_drop
# Bu sütunları veri setinden çıkarma
df_cleaned = df.drop(columns=drop_columns)

print("\nTüm Değerleri Aynı ve Farklı Olan Sütunlar Çıkarıldı:")
print(df_cleaned)

