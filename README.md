# Sigorta Maliyeti Tahmin Uygulaması
# Insurance Cost Prediction Application

## TUR 🇹🇷 | ENG 🇺🇸

Kaggle üzerindeki insurance.csv veri seti ile Random Forest kullanarak sigorta maliyeti tahmini yapan bir masaüstü uygulaması.
A desktop application that predicts insurance costs using Random Forest with the insurance.csv dataset from Kaggle.

## Özellikler | Features

- Random Forest Regressor modeli | Random Forest Regressor model
    - 0.87 R2 score
- Otomatik veri önişleme | Automatic data preprocessing:
  - Kategorik verileri one-hot encoding | Categorical data one-hot encoding
  - Sayısal verileri standardizasyon | Numerical data standardization
- PyQt5 ile arayüz | PyQt5 interface
- Eğitilmiş modeli kaydetme ve yükleme özelliği | Save and load trained model feature

## Kurulum | Installation

1. **Gereksinimleri yükleyin | Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Modeli eğitin | Train the model:**
   ```bash
   python save_model.py
   ```

3. **Uygulamayı başlatın | Start the application:**
   ```bash
   python app.py
   ```

## Proje Yapısı | Project Structure

```
insurance_prediction_app/
├── app.py                # Ana uygulama | Main application
├── model.py              # Model eğitimi ve değerlendirme | Model training and evaluation
├── insurance_gui.py      # Arayüz dosyası | Interface
├── save_model.py         # Model kaydetme | Model saving
├── insurance.csv         # Veri seti | Dataset
└── models/               # Kaydedilmiş model dosyaları | Saved model files
    ├── insurance_model.pkl
    ├── feature_encoder.pkl
    ├── feature_scaler.pkl
    └── feature_names.pkl
```

## Teknolojiler | Technologies

- Python 3.12.1
- PyQt5
- scikit-learn
- pandas
- numpy