# Prediksi Hasil Pertandingan Liga Inggris Menggunakan XGBoost

Proyek ini menggunakan algoritma XGBoost untuk memprediksi hasil pertandingan Liga Inggris (Premier League) dengan berbagai model dan pendekatan.

## Struktur Proyek

Proyek ini terdiri dari beberapa direktori utama:

- `data/`: Berisi dataset pertandingan Liga Inggris
  - `Dataset EPL New/`: Dataset dengan odds dan fitur lengkap
  - `No Odds Dataset/`: Dataset tanpa odds untuk prediksi realistis

- `historical_prediction/`: Skrip untuk prediksi berbasis data historis
  - `train_xgboost_with_odds.py`: Melatih model XGBoost dengan odds
  - `cross_validate_model.py`: Validasi silang untuk model
  - `test_xgboost_model.py`: Menguji model pada data baru

- `future_prediction/`: Skrip untuk prediksi pertandingan masa depan
  - `generate_prematch_features.py`: Menghasilkan fitur prematch
  - `train_xgboost_realistis.py`: Melatih model XGBoost tanpa odds
  - `train_xgboost_prematch.py`: Melatih model XGBoost dengan fitur prematch
  - `predict_new_matches.py`: Memprediksi pertandingan baru

- `models/`: Menyimpan model XGBoost terlatih
- `predictions/`: Menyimpan hasil prediksi dan visualisasi

## Jenis Model

Proyek ini menggunakan tiga jenis model XGBoost:

1. **Model dengan Odds** - Menggunakan odds dari bandar taruhan untuk prediksi akurat (akurasi ~97%)
2. **Model Realistis** - Menggunakan statistik tim tanpa odds (akurasi ~52%)
3. **Model Prematch** - Menggunakan fitur prematch yang dihitung dari data historis (akurasi ~62%)

## Fitur-fitur

Beberapa fitur yang digunakan dalam model:

- Statistik tim (rata-rata gol, kebobolan, dll)
- Form tim (hasil beberapa pertandingan terakhir)
- Statistik head-to-head
- Performa kandang dan tandang
- Odds (untuk model dengan odds)

## Cara Penggunaan

### 1. Melatih Model

Untuk melatih model dengan odds:
```
python historical_prediction/train_xgboost_with_odds.py
```

Untuk melatih model realistis tanpa odds:
```
python future_prediction/train_xgboost_realistis.py
```

Untuk melatih model dengan fitur prematch:
```
python future_prediction/generate_prematch_features.py
python future_prediction/train_xgboost_prematch.py
```

### 2. Memprediksi Pertandingan Baru

Untuk memprediksi pertandingan baru:
```
python future_prediction/predict_new_matches.py
```

Hasil prediksi akan disimpan di direktori `predictions/`.

## Performa Model

- **Model dengan Odds**: Akurasi ~97%
- **Model Realistis**: Akurasi ~52%
- **Model Prematch**: Akurasi ~62%

## Visualisasi

Proyek ini menghasilkan beberapa visualisasi:
- Confusion matrix untuk evaluasi model
- Feature importance untuk melihat fitur yang paling berpengaruh
- Grafik probabilitas untuk setiap prediksi pertandingan

## Persyaratan

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib
- tqdm

## Catatan

Model prediksi sepak bola memiliki keterbatasan karena sifat olahraga yang tidak dapat diprediksi sepenuhnya. Model dengan odds cenderung lebih akurat karena odds mencerminkan banyak faktor yang tidak tertangkap dalam data statistik sederhana. 