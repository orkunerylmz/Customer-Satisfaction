import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#
pd.options.display.max_rows = 9999

df = pd.read_csv("data.csv")

# Eksik verilerin doldurulması
columns = ['Income', 'ProductQuality', 'ServiceQuality', 'PurchaseFrequency', 'SatisfactionScore']
for col in columns:
    df[col] = df[col].fillna(df[col].median())


# Eksik verilerin doldurulması sonrasında eksik veri kontrolü
missing_data = df.isna().sum().sum()  # Eksik veri sayısı 
print(f"Eksik Veri Sayısı = {missing_data}")

# Outlier tespiti ve temizleme
df = df[df['Age'] < 100] # 100 yaş üstü veri dışlanıyor
df = df[df['Gender'] != ("Female", "Male")] # Farkli cinsiyet içeren veri dışlanıyor.
df = df[df['SatisfactionScore'] <= 100] # Memnuniyet skoru 100 puan üstü veri dışlanıyor.
df = df[df['SatisfactionScore'] >= 0] # Memnuniyet skoru 0 puan altı veri dışlanıyor.

print(df.head(1219))

# Yeni temiz veri dosya içerisine aktarılır.
df.to_csv("clean_data.csv", index = False, encoding = "utf-8")

# Müşteri yaşı ve memnuniyet ilişkisi
plt.figure(1)
sns.scatterplot(x='SatisfactionScore', y='Age', data=df)
plt.title('Müşteri Yaşı ve Memnuniyet İlişkisi')

# Satın Alma Sıklığı ve Sadakat Seviyesi
plt.figure(2)
sns.boxplot(x='LoyaltyLevel', y='PurchaseFrequency', data=df)
plt.title('Satın Alma Sıklığı ve Sadakat Seviyesi')

# Ürün Kalitesi ve Geri Bildirim Skoru
plt.figure(3)
sns.boxplot(x='ProductQuality', y='FeedbackScore', data=df)
plt.title('Ürün Kalitesi ve Geri Bildirim Skoru')

# Gelir ve Satın Alma Sıklığı
plt.figure(4)
sns.kdeplot(x='PurchaseFrequency', y='Income', data=df, cmap='Reds', fill=True)
plt.title('Gelir ve Satın Alma Sıklığı')

plt.tight_layout()  # Grafiklerin üst üste binmesini engeller.

plt.show()  # Grafik gosterme


# Veri setini bölme
X = df[['Age', 'Income', 'ProductQuality']]  # Bağımsız değişkenler
y = df['SatisfactionScore'].apply(lambda x: 1 if x > 50 else 0)  # Bağımlı değişken (sınıflandırma)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model oluşturma
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahmin ve doğrulama
y_pred = model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
