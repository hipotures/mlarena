# Analiza kolumn kategorycznych zakodowanych jako integery
**Dataset:** Playground Series S5E12 (Diabetes Prediction)
**Data analizy:** 2025-12-08

## Podsumowanie wykonawcze

W zbiorze danych `playground-series-s5e12` zidentyfikowano **5 kolumn**, które reprezentują zmienne kategoryczne mimo kodowania jako liczby całkowite (int64/float64). Narzędzia analityczne takie jak YData Profiling błędnie interpretują te kolumny jako zmienne numeryczne, co prowadzi do nieprawidłowej analizy i potencjalnie gorszych wyników modelowania.

---

## Zidentyfikowane kolumny kategoryczne

### 1. Kolumny binarne (0/1)

#### `family_history_diabetes`
- **Typ danych:** int64
- **Wartości:** 0, 1
- **Znaczenie:** Historia cukrzycy w rodzinie (0 = nie, 1 = tak)
- **Rozkład:**
  - 0: 595,419 (85.1%)
  - 1: 104,581 (14.9%)

#### `hypertension_history`
- **Typ danych:** int64
- **Wartości:** 0, 1
- **Znaczenie:** Historia nadciśnienia (0 = nie, 1 = tak)
- **Rozkład:**
  - 0: 572,607 (81.8%)
  - 1: 127,393 (18.2%)

#### `cardiovascular_history`
- **Typ danych:** int64
- **Wartości:** 0, 1
- **Znaczenie:** Historia chorób sercowo-naczyniowych (0 = nie, 1 = tak)
- **Rozkład:**
  - 0: 678,773 (97.0%)
  - 1: 21,227 (3.0%)

#### `diagnosed_diabetes` (TARGET)
- **Typ danych:** float64 (train.csv) / int64 (orig)
- **Wartości:** 0.0, 1.0
- **Znaczenie:** Zdiagnozowana cukrzyca (0 = nie, 1 = tak)
- **Rozkład:**
  - 0.0: 263,693 (37.7%)
  - 1.0: 436,307 (62.3%)

### 2. Kolumny porządkowe (ordinal)

#### `alcohol_consumption_per_week`
- **Typ danych:** int64
- **Wartości:** 1, 2, 3, 4, 5, 6, 7, 8, 9 (train) | 0-10 (orig)
- **Znaczenie:** Tygodniowe spożycie alkoholu (prawdopodobnie skala kategoryczna)
- **Rozkład:**
  - 1: 246,311 (35.2%)
  - 2: 246,592 (35.2%)
  - 3: 137,565 (19.7%)
  - 4: 52,973 (7.6%)
  - 5: 13,322 (1.9%)
  - 6: 2,728 (0.4%)
  - 7: 447 (0.06%)
  - 8: 59 (0.01%)
  - 9: 3 (0.0004%)

**Uwaga:** Silna asymetria rozkładu (większość wartości 1-3) sugeruje, że to może być zakodowana zmienna kategoryczna reprezentująca poziomy spożycia (np. "niskie", "średnie", "wysokie").

---

## Porównanie: Train vs Original Dataset

| Kolumna | Train (synthetic) | Original | Różnice |
|---------|-------------------|----------|---------|
| `family_history_diabetes` | int64, 2 wartości | int64, 2 wartości | ✓ Zgodne |
| `hypertension_history` | int64, 2 wartości | int64, 2 wartości | ✓ Zgodne |
| `cardiovascular_history` | int64, 2 wartości | int64, 2 wartości | ✓ Zgodne |
| `diagnosed_diabetes` | float64, 2 wartości | int64, 2 wartości | ⚠ Typ różny |
| `alcohol_consumption_per_week` | int64, 9 wartości (1-9) | int64, 11 wartości (0-10) | ⚠ Zakres różny |

---

## Rekomendacje dla narzędzi analitycznych

### YData Profiling / pandas-profiling
```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('train.csv')

# Konwersja na kategorie PRZED analizą
categorical_cols = [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'diagnosed_diabetes',
    'alcohol_consumption_per_week'
]

for col in categorical_cols:
    df[col] = df[col].astype('category')

profile = ProfileReport(df, title="S5E12 Diabetes - Categorical Fixed")
profile.to_file("diabetes_profile_fixed.html")
```

### XGBoost / LightGBM / CatBoost
```python
# XGBoost: automatyczne wykrywanie
dtrain = xgb.DMatrix(
    X_train,
    label=y_train,
    enable_categorical=True,
    feature_types=['c'] * len(categorical_cols) + ['q'] * len(numeric_cols)
)

# LightGBM: parametr categorical_feature
lgb.train(
    params,
    train_data,
    categorical_feature=['family_history_diabetes', 'hypertension_history', ...]
)

# CatBoost: automatyczne wykrywanie int jako kategorii
cat_features = [0, 1, 2, 3, 4]  # indeksy kolumn kategorycznych
model = CatBoostClassifier(cat_features=cat_features)
```

### AutoGluon
```python
# Automatyczne wykrywanie typów - wymaga konwersji przed trenowaniem
df[categorical_cols] = df[categorical_cols].astype('category')

predictor = TabularPredictor(label='diagnosed_diabetes')
predictor.fit(df)
```

---

## Dane statystyczne

### Dataset: train.csv (synthetic)
- **Liczba rekordów:** 700,000
- **Liczba kolumn:** 26
- **Kolumny kategoryczne (string):** 6 (gender, ethnicity, education_level, income_level, smoking_status, employment_status)
- **Kolumny kategoryczne (int):** 5 (jak powyżej)
- **Kolumny numeryczne (continuous):** 15

### Dataset: diabetes_dataset.csv (original)
- **Liczba rekordów:** 100,000
- **Liczba kolumn:** 31
- **Kolumny kategoryczne (int):** 5 (jak powyżej)
- **Dodatkowe kolumny (vs train):** glucose_fasting, glucose_postprandial, insulin_level, hba1c, diabetes_risk_score, + 1 extra

---

## Uzasadnienie klasyfikacji

### Dlaczego te kolumny są kategoryczne?

1. **Binarne 0/1 (medical history):** Reprezentują obecność/brak historii choroby. To klasyczne zmienne binarne kategoryczne, nie wartości numeryczne do obliczeń matematycznych.

2. **alcohol_consumption_per_week:** Mimo numerycznego kodowania (1-9):
   - Silnie dyskretny rozkład
   - Nieliniowa relacja z targetem (prawdopodobnie)
   - Wartości prawdopodobnie reprezentują przedziały/kategorie (np. "1 drink", "2-3 drinks", etc.)
   - W kontekście medycznym często kodowane jako porządkowe kategorie

3. **diagnosed_diabetes (target):** Mimo że to target, jest to binarna zmienna kategoryczna (klasyfikacja binarna, nie regresja).

---

## Przykład: Fragment danych

```csv
id,family_history_diabetes,hypertension_history,cardiovascular_history,alcohol_consumption_per_week,diagnosed_diabetes
0,0,0,0,1,1.0
1,0,0,0,2,1.0
2,0,0,0,3,0.0
3,0,1,0,3,1.0
```

**Interpretacja:**
- Rekord 0: Brak historii rodzinnej, brak nadciśnienia, brak chorób CV, niskie spożycie alkoholu → Zdiagnozowana cukrzyca
- Rekord 3: Brak historii rodzinnej, **ma nadciśnienie**, brak chorób CV, średnie spożycie → Zdiagnozowana cukrzyca

---

## Prompt dla modelu AI (do wykorzystania)

```
Dataset: Playground Series S5E12 - Diabetes Prediction
Rekordów: 700,000 | Kolumn: 26

Problem: Narzędzia analityczne (YData Profiling) błędnie traktują niektóre kolumny
jako numeryczne, mimo że reprezentują zmienne kategoryczne.

Kolumny do weryfikacji (zakodowane jako int64/float64):
1. family_history_diabetes: 0/1
2. hypertension_history: 0/1
3. cardiovascular_history: 0/1
4. diagnosed_diabetes: 0.0/1.0 (TARGET)
5. alcohol_consumption_per_week: 1-9

Przykładowe rekordy:
id,age,alcohol_consumption_per_week,family_history_diabetes,hypertension_history,cardiovascular_history,diagnosed_diabetes
0,31,1,0,0,0,1.0
1,50,2,0,0,0,1.0
2,32,3,0,0,0,0.0
3,54,3,0,1,0,1.0

Pytanie: Które z tych kolumn powinny być traktowane jako kategoryczne
(binarne lub porządkowe) zamiast jako zmienne numeryczne ciągłe?
```

---

## Kontekst konkursu Kaggle

**Źródło:** https://www.kaggle.com/competitions/playground-series-s5e12
**Typ:** Classification (Binary)
**Metric:** ROC AUC
**Original dataset:** Diabetes Health Indicators Dataset (100k samples)
**Synthetic dataset:** 700k samples wygenerowanych na podstawie oryginału

**Kluczowe wskazówki z opisu konkursu:**
- Problem klasyfikacji binarnej (przewidywanie cukrzycy)
- Dane medyczne/zdrowotne
- Zawierają informacje o historii chorób (naturalne zmienne binarne)
- Wiele cech behawioralnych (dieta, alkohol, aktywność) często kodowanych jako kategorie

---

## Wnioski

1. **5 kolumn** wymaga traktowania jako kategoryczne mimo kodowania numerycznego
2. **4 kolumny binarne** (0/1): medical history features + target
3. **1 kolumna porządkowa**: alcohol consumption (dyskretna skala)
4. Błędna klasyfikacja prowadzi do:
   - Nieprawidłowych statystyk (mean, std dla kategorii)
   - Błędnych korelacji
   - Gorszej jakości feature engineering
   - Suboptymalne wyniki modelowania

**Zalecenie:** Zawsze konwertuj te kolumny na typ `category` przed analizą i modelowaniem.
