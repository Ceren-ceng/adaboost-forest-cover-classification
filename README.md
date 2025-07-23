
# 🌲 AdaBoost ile Orman Örtüsü Türü Sınıflandırması

Bu projede **ensemble learning** yöntemlerinden biri olan **AdaBoost (Adaptive Boosting)** algoritmasını kullanarak, `covtype.csv` veri seti üzerinden orman örtüsü türlerini sınıflandırdık.

---

## 📌 1. Ensemble Learning Nedir?

Ensemble Learning, birden fazla zayıf modelin (örneğin tek başına pek başarılı olmayan ağaçların) bir araya gelerek güçlü bir tahmin modeli oluşturmasını sağlar. Bu teknik üç ana gruba ayrılır:

- **Bagging** → Örnek: Random Forest 🌲
- **Boosting** → Örnek: AdaBoost ⚡
- **Stacking** → Farklı modellerin çıktıları birleştirilerek son model eğitilir.

Bu projede odaklandığımız yöntem **Boosting**, özellikle de **AdaBoost** algoritmasıdır.

---

## ⚡ 2. AdaBoost Nedir?

**AdaBoost (Adaptive Boosting)**, karar ağaçları gibi zayıf sınıflandırıcıları art arda eğitip birleştirerek güçlü bir model üretir.

🔁 Her adımda model hatalı sınıflandırdığı örneklere daha fazla ağırlık verir.  
🎯 Amaç: Zayıf öğrenicileri zincirleme kullanarak hataları azaltmak.  
🌳 Genellikle temel öğrenici olarak **DecisionTreeClassifier (max_depth=1)** kullanılır.

---

## ❓ Neden AdaBoost Kullandık?

- Orman örtüsü gibi **karmaşık ve çok boyutlu** verilerde, doğrusal olmayan ilişkileri öğrenmekte çok başarılı.
- Tek başına karar ağacı modellerinden **daha yüksek doğruluk sağlar**.
- Boosting'in doğası gereği **ağırlıklandırılmış öğrenme** sayesinde, zor örneklerde daha isabetli tahminler yapılabilir.

---

## 📊 3. Kullanılan Veri Seti: `covtype.csv`

- Veri Kümesi: 581.012 satır, 55 sütun
- Amaç: Son sütun olan `Cover_Type`'ı tahmin etmek
- Özellikler:
  - Sayısal veriler: `Elevation`, `Slope`, `Hillshade`, `Distance` sütunları
  - One-hot encoded veriler: `Wilderness_Area` (4 kolon), `Soil_Type` (40 kolon)

📁 Veri kümesi Kaggle’dan alınmıştır ve yaygın olarak sınıflandırma problemlerinde kullanılır.

---

## 🔧 4. Kullanılan Kütüphaneler ve Sebepleri

- `pandas`: Veri yükleme ve işleme için
- `numpy`: Sayısal işlemler için
- `sklearn.model_selection.train_test_split`: Eğitim/test bölme
- `sklearn.preprocessing.StandardScaler`: Ölçekleme
- `sklearn.ensemble.AdaBoostClassifier`: Ana model
- `sklearn.tree.DecisionTreeClassifier`: Temel sınıflandırıcı
- `sklearn.metrics`: Accuracy, classification_report, confusion matrix gibi metrikler

---

## 🔍 5. Model Yapısı

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=1.0)
```

### Açıklama:
- `base_estimator`: Her bir adımda kullanılan karar ağacı modeli. `max_depth=1` seçildi çünkü boosting algoritmaları zayıf öğreniciler ile çalışır.
- `n_estimators`: Toplam kaç zayıf model oluşturulacak?
- `learning_rate`: Her zayıf modelin etkisini belirler.

---

## 📈 6. Model Performansı

| Metrik                 | Değer |
|------------------------|-------|
| Doğruluk (Accuracy)    | %83   |
| Precision, Recall, F1  | Sınıflara göre detaylı raporlandı
| Confusion Matrix       | 🔲 🔳 🔲 🔳 (yüksek başarı, bazı sınıflarda karışıklık)

📌 Not: AdaBoost modeli özellikle belirli sınıflarda (örneğin 1 ve 2. türlerde) çok başarılı tahminler yaptı. Ancak bazı benzer sınıflar arasında (örneğin 4 ve 5) karışmalar gözlemlendi.

---

## ✅ AdaBoost’un Avantajları

- Hatalı sınıflandırmalardan öğrenir 🔁
- Ağırlıklandırma ile zor örneklerde gelişme sağlar
- Diğer yöntemlere kıyasla çok büyük veri setlerinde bile hızlı çalışır

---

## ⚠️ Dikkat Edilmesi Gerekenler

- Outlier'lara karşı duyarlıdır 🚨
- Overfitting riski yok denemez, ama genelde düşüktür
- Her zaman daha iyi demek değil, veri setine göre model seçilmelidir

---

## 📎 Projeyi Çalıştırmak İçin

1. Python 3.8+ yüklü olmalı
2. Gerekli kütüphaneleri yükle:
```bash
pip install -r requirements.txt
```
3. Notebook'u çalıştır: `adaboost.ipynb`
4. Sonuçları gözlemle 📊

---

## 🧠 Bonus: AdaBoost vs. Diğer Yöntemler

| Özellik        | AdaBoost     | Random Forest |
|----------------|--------------|----------------|
| Model türü     | Boosting     | Bagging        |
| Öğrenme tipi   | Ağırlıklı    | Rastgele örnekleme |
| Hedef          | Hatalı örneklerden ders alma | Overfitting’i azaltma |
| Base Learner   | Zayıf (depth=1) karar ağacı | Karar ağaçları |

---

## ✍️ Katkı ve Lisans

Bu proje eğitim amaçlıdır. Her türlü öneri, katkı ve yıldız ⭐ memnuniyetle karşılanır.
