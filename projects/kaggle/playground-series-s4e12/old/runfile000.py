import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, PolynomialFeatures
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from sklearn.impute import SimpleImputer
import optuna
from itertools import combinations
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Definicja kroków
total_steps = 22
current_step = 0

def print_step(description):
    global current_step
    current_step += 1
    print(f"Step {current_step}/{total_steps}: {description}")

# Krok 1: Import bibliotek
print_step("Importowanie bibliotek")

# Krok 2: Wczytanie danych
print_step("Wczytywanie danych treningowych i testowych")
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Krok 3: Sampling danych treningowych
print_step("Losowanie 10% danych treningowych i resetowanie indeksu")
train = train.sample(frac=0.1, random_state=42).reset_index(drop=True)

# Krok 4: Oddzielenie cech i celu
print_step("Oddzielanie cech od celu")
X = train.drop(["Premium Amount", "id"], axis=1)
y = np.log1p(train["Premium Amount"])
test_features = test.drop("id", axis=1)

# Krok 5: Identyfikacja kolumn kategorycznych i numerycznych
print_step("Identyfikacja kolumn kategorycznych i numerycznych")
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Krok 6: Inicjalizacja imputerów
print_step("Inicjalizacja imputerów dla brakujących wartości")
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Krok 7: Imputacja brakujących wartości
print_step("Imputacja brakujących wartości w danych treningowych i testowych")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test_features[num_cols] = num_imputer.transform(test_features[num_cols])
test_features[cat_cols] = cat_imputer.transform(test_features[cat_cols])

# Krok 8: Tworzenie cech agregacyjnych
print_step("Tworzenie cech agregacyjnych")
agg_features = []
agg_feature_names = []
mean_features = []
agg_stats_dict = {}

important_num_cols = ["Age", "Annual Income", "Health Score"]

for cat_col in cat_cols:
    for num_col in important_num_cols:
        agg_key = f"{cat_col}_{num_col}"
        if agg_key not in agg_stats_dict:
            agg_stats = X.groupby(cat_col)[num_col].agg(["mean", "std", "min", "max"]).reset_index()
            agg_stats.columns = [cat_col] + [f"{agg_key}_{stat}" for stat in ["mean", "std", "min", "max"]]
            agg_stats_dict[agg_key] = agg_stats
        
        # Merge agregacji do danych treningowych
        X = X.merge(agg_stats_dict[agg_key], on=cat_col, how="left")
        # Merge agregacji do danych testowych
        test_features = test_features.merge(agg_stats_dict[agg_key], on=cat_col, how="left")
        
        # Zbieranie nazw cech
        mean_feature_name = f"{agg_key}_mean"
        if mean_feature_name not in mean_features:
            mean_features.append(mean_feature_name)
        agg_feature_names.extend([f"{agg_key}_{stat}" for stat in ["mean", "std", "min", "max"]])

# Krok 9: Tworzenie cech interakcyjnych między średnimi agregacji
print_step("Tworzenie cech interakcyjnych między średnimi agregacji")
interaction_features = []
for feat1, feat2 in combinations(mean_features, 2):
    interaction_name = f"interaction_{feat1}_{feat2}"
    X[interaction_name] = X[feat1] * X[feat2]
    test_features[interaction_name] = test_features[feat1] * test_features[feat2]
    interaction_features.append(interaction_name)
    agg_feature_names.append(interaction_name)

# Krok 10: Tworzenie cech binned (grupowanych)
print_step("Tworzenie cech binowanych dla ważnych kolumn numerycznych")
binner = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile")
X_binned = binner.fit_transform(X[important_num_cols])
test_binned = binner.transform(test_features[important_num_cols])

binned_feature_names = [
    f"{col}_bin_{i}" for col in important_num_cols for i in range(10)
]
X_binned_df = pd.DataFrame(X_binned, columns=binned_feature_names, index=X.index)
test_binned_df = pd.DataFrame(test_binned, columns=binned_feature_names, index=test_features.index)

# Krok 11: Tworzenie cech polinomialnych
print_step("Tworzenie cech polinomialnych")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[important_num_cols])
test_poly = poly.transform(test_features[important_num_cols])
poly_feature_names = poly.get_feature_names_out(important_num_cols)

# Krok 12: Kodowanie cech kategorycznych za pomocą Target Encoding
print_step("Target Encoding dla cech kategorycznych")
te = TargetEncoder()
X_cat_encoded = te.fit_transform(X[cat_cols], y)
test_cat_encoded = te.transform(test_features[cat_cols])

# Krok 13: Skalowanie cech numerycznych
print_step("Skalowanie cech numerycznych")
scaler_num = StandardScaler()
X_num_scaled = pd.DataFrame(scaler_num.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
test_num_scaled = pd.DataFrame(
    scaler_num.transform(test_features[num_cols]), columns=num_cols, index=test_features.index
)

# Krok 14: Skalowanie cech polinomialnych (tylko terminy nieliniowe)
print_step("Skalowanie cech polinomialnych (tylko terminy nieliniowe)")
# Zidentyfikowanie cech nieliniowych (interakcje i kwadraty)
poly_non_linear_indices = [i for i, name in enumerate(poly_feature_names) if '*' in name]
X_poly_non_linear = X_poly[:, poly_non_linear_indices]
test_poly_non_linear = test_poly[:, poly_non_linear_indices]

scaler_poly = StandardScaler()
X_poly_scaled = pd.DataFrame(
    scaler_poly.fit_transform(X_poly_non_linear),
    columns=[f"poly_{name}" for name in np.array(poly_feature_names)[poly_non_linear_indices]],
    index=X.index
)
test_poly_scaled = pd.DataFrame(
    scaler_poly.transform(test_poly_non_linear),
    columns=[f"poly_{name}" for name in np.array(poly_feature_names)[poly_non_linear_indices]],
    index=test_features.index
)

# Krok 15: Skalowanie cech agregacyjnych
print_step("Skalowanie cech agregacyjnych")
scaler_agg = StandardScaler()
X_agg_scaled = pd.DataFrame(
    scaler_agg.fit_transform(X[agg_feature_names]),
    columns=agg_feature_names,
    index=X.index
)
test_agg_scaled = pd.DataFrame(
    scaler_agg.transform(test_features[agg_feature_names]),
    columns=agg_feature_names,
    index=test_features.index
)

# Krok 16: Łączenie wszystkich przetworzonych cech
print_step("Łączenie wszystkich przetworzonych cech")
X_processed = pd.concat(
    [X_num_scaled, X_cat_encoded, X_poly_scaled, X_binned_df, X_agg_scaled], axis=1
)
test_processed = pd.concat(
    [test_num_scaled, test_cat_encoded, test_poly_scaled, test_binned_df, test_agg_scaled],
    axis=1,
)

# Krok 17: Definicja funkcji celu dla Optuna
print_step("Definiowanie funkcji celu dla Optuna")

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "bagging_freq": 5,
        "random_state": 42,
        "n_jobs": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_processed):
        X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
            verbose_eval=False
        )

        val_pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
        y_val_orig = np.expm1(y_val)

        fold_score = np.sqrt(mean_squared_log_error(y_val_orig, val_pred))
        cv_scores.append(fold_score)

    return np.mean(cv_scores)

# Krok 18: Optymalizacja hiperparametrów za pomocą Optuna
print_step("Uruchamianie optymalizacji hiperparametrów za pomocą Optuna")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Krok 19: Pobranie najlepszych parametrów
print_step("Pobieranie najlepszych parametrów z Optuna")
best_params = study.best_params
best_params.update(
    {
        "objective": "regression",
        "metric": "rmse",
        "bagging_freq": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
)

# Krok 20: Trenowanie końcowego modelu z najlepszymi parametrami
print_step("Trenowanie końcowego modelu z najlepszymi parametrami")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
test_predictions = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed), 1):
    print(f"  Fold {fold}/5: Trenowanie modelu")
    X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
        verbose_eval=False
    )

    val_pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
    y_val_orig = np.expm1(y_val)

    fold_score = np.sqrt(mean_squared_log_error(y_val_orig, val_pred))
    cv_scores.append(fold_score)
    print(f"  Fold {fold}/5: RMSLE = {fold_score:.4f}")

    test_pred = np.expm1(model.predict(test_processed, num_iteration=model.best_iteration))
    test_predictions.append(test_pred)

# Krok 21: Wyświetlenie wyników walidacji krzyżowej
print_step("Wyświetlanie wyników walidacji krzyżowej")
print(f"Average RMSLE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Krok 22: Generowanie pliku z prognozami
print_step("Generowanie pliku z prognozami")
final_predictions = np.mean(test_predictions, axis=0)
submission = pd.DataFrame({"id": test["id"], "Premium Amount": final_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Pipeline zakończony pomyślnie. Plik submission.csv został zapisany.")

