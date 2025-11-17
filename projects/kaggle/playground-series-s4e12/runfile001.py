import os  # Added for file and path operations
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

warnings.filterwarnings("ignore")

print("Step 1/22: Importowanie bibliotek")

print("Step 2/22: Wczytywanie danych treningowych i testowych")
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

print("Step 3/22: Losowanie 10% danych treningowych i resetowanie indeksu")
train = train.sample(frac=1.0, random_state=43).reset_index(drop=True)

print("Step 4/22: Oddzielanie cech od celu")
X = train.drop(["Premium Amount", "id"], axis=1)
y = np.log1p(train["Premium Amount"])
test_features = test.drop("id", axis=1)

print("Step 5/22: Identyfikacja kolumn kategorycznych i numerycznych")
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

print("Step 6/22: Inicjalizacja imputerów dla brakujących wartości")
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

print("Step 7/22: Imputacja brakujących wartości w danych treningowych i testowych")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test_features[num_cols] = num_imputer.transform(test_features[num_cols])
test_features[cat_cols] = cat_imputer.transform(test_features[cat_cols])

print("Step 8/22: Tworzenie cech agregacyjnych, w tym EWMA")
agg_features_train = []
agg_features_test = []
mean_features = []
agg_feature_names = []
decay_factors = [0.3, 0.5, 0.7]
important_num_cols = ["Age", "Annual Income", "Health Score"]

for cat_col in cat_cols:
    for num_col in important_num_cols:
        agg_stats = X.groupby(cat_col)[num_col].agg(["mean", "std", "min", "max"]).reset_index()
        agg_stats.columns = [cat_col] + [f"{cat_col}_{num_col}_{stat}" for stat in ["mean", "std", "min", "max"]]
        agg_features_train.append(X.merge(agg_stats, on=cat_col, how="left").iloc[:, -4:])
        agg_features_test.append(test_features.merge(agg_stats, on=cat_col, how="left").iloc[:, -4:])
        
        mean_feature_name = f"{cat_col}_{num_col}_mean"
        mean_features.append(mean_feature_name)
        agg_feature_names.extend([f"{cat_col}_{num_col}_{stat}" for stat in ["mean", "std", "min", "max"]])
        
        for alpha in decay_factors:
            ewma_name = f"{cat_col}_{num_col}_ewma_{alpha}"
            ewma_train = X.groupby(cat_col)[num_col].transform(lambda x: x.ewm(alpha=alpha).mean().shift(1))
            ewma_test = test_features.groupby(cat_col)[num_col].transform(lambda x: x.ewm(alpha=alpha).mean().shift(1))
            agg_features_train.append(ewma_train.rename(ewma_name))
            agg_features_test.append(ewma_test.rename(ewma_name))
            agg_feature_names.append(ewma_name)

agg_features_train_df = pd.concat([df if isinstance(df, pd.DataFrame) else df.to_frame() for df in agg_features_train], axis=1)
agg_features_test_df = pd.concat([df if isinstance(df, pd.DataFrame) else df.to_frame() for df in agg_features_test], axis=1)

X = pd.concat([X, agg_features_train_df], axis=1)
test_features = pd.concat([test_features, agg_features_test_df], axis=1)

print("Step 9/22: Tworzenie cech interakcyjnych między średnimi agregacji")
interaction_features_train = []
interaction_features_test = []

for feat1, feat2 in combinations(mean_features, 2):
    interaction_name = f"interaction_{feat1}_{feat2}"
    interaction_features_train.append(X[feat1] * X[feat2])
    interaction_features_test.append(test_features[feat1] * test_features[feat2])
    agg_feature_names.append(interaction_name)

interaction_features_train_df = pd.concat(
    [df.rename(f"interaction_{feat1}_{feat2}") for df, (feat1, feat2) in zip(interaction_features_train, combinations(mean_features, 2))],
    axis=1
)
interaction_features_test_df = pd.concat(
    [df.rename(f"interaction_{feat1}_{feat2}") for df, (feat1, feat2) in zip(interaction_features_test, combinations(mean_features, 2))],
    axis=1
)

X = pd.concat([X, interaction_features_train_df], axis=1)
test_features = pd.concat([test_features, interaction_features_test_df], axis=1)

print("Step 10/22: Tworzenie cech binowanych dla ważnych kolumn numerycznych")
binner = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile")
X_binned = binner.fit_transform(X[important_num_cols])
test_binned = binner.transform(test_features[important_num_cols])

binned_feature_names = [f"{col}_bin_{i}" for col in important_num_cols for i in range(10)]
X_binned_df = pd.DataFrame(X_binned, columns=binned_feature_names, index=X.index)
test_binned_df = pd.DataFrame(test_binned, columns=binned_feature_names, index=test_features.index)

print("Step 11/22: Tworzenie cech polinomialnych")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[important_num_cols])
test_poly = poly.transform(test_features[important_num_cols])
poly_feature_names = poly.get_feature_names_out(important_num_cols)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
test_poly_df = pd.DataFrame(test_poly, columns=poly_feature_names, index=test_features.index)

print("Step 12/22: Target Encoding dla cech kategorycznych")
te = TargetEncoder()
X_cat_encoded = te.fit_transform(X[cat_cols], y)
test_cat_encoded = te.transform(test_features[cat_cols])

print("Step 13/22: Skalowanie cech numerycznych")
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
test_num_scaled = pd.DataFrame(scaler.transform(test_features[num_cols]), columns=num_cols, index=test_features.index)

print("Step 14/22: Dodawanie cech polinomialnych")
# X_poly_df already created

print("Step 15/22: Skalowanie cech agregacyjnych")
scaler_agg = StandardScaler()
X_agg = X[agg_feature_names]
test_agg = test_features[agg_feature_names]
X_agg_scaled = pd.DataFrame(scaler_agg.fit_transform(X_agg), columns=agg_feature_names, index=X.index)
test_agg_scaled = pd.DataFrame(scaler_agg.transform(test_agg), columns=agg_feature_names, index=test_features.index)

print("Step 16/22: Łączenie wszystkich cech")
X_processed = pd.concat([X_num_scaled, X_cat_encoded, X_poly_df, X_binned_df, X_agg_scaled], axis=1)
test_processed = pd.concat([test_num_scaled, test_cat_encoded, test_poly_df, test_binned_df, test_agg_scaled], axis=1)

X_processed = X_processed.loc[:, ~X_processed.columns.duplicated()]
test_processed = test_processed.loc[:, ~test_processed.columns.duplicated()]

print("Step 17/22: Definicja funkcji celu dla Optuna")
def objective(trial):
    params = {
        "device": "gpu",
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "bagging_freq": 5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_processed):
        X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, params={'verbose': -1}, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, params={'verbose': -1}, free_raw_data=False)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        )

        val_pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
        y_val_orig = np.expm1(y_val)

        fold_score = np.sqrt(mean_squared_log_error(y_val_orig, val_pred))
        cv_scores.append(fold_score)

    return np.mean(cv_scores)


# Dynamicne pobieranie nazwy skryptu bez rozszerzenia .py
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Ścieżka do pliku journal
journal_path = './optuna-study.journal'

# Sprawdzenie, czy plik journal istnieje
if os.path.exists(journal_path):
    LOAD_STUDY = True
else:
    LOAD_STUDY = False

# Ustawienie przechowywania studia
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(journal_path),
)

# Ustawienie logowania Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

# Ładowanie lub tworzenie studia
if LOAD_STUDY:
    try:
        study = optuna.load_study(
            study_name=script_name, 
            storage=storage,
        )
        print("Study loaded")
    except KeyError:
        # Jeśli study o danej nazwie nie istnieje w journal, tworzymy nowe
        study = optuna.create_study(
            direction='minimize', 
            pruner=optuna.pruners.MedianPruner(), 
            storage=storage, 
            study_name=script_name
        )
        print("Study created (KeyError: Study not found, created new)")
else:
    study = optuna.create_study(
        direction='minimize', 
        pruner=optuna.pruners.MedianPruner(), 
        storage=storage, 
        study_name=script_name
    )
    print("Study created")

# Optymalizacja hyperparametrów za pomocą Optuna w pętli
print("Step 18/22: Optymalizacja hyperparametrów za pomocą Optuna")

while True: 
    study.optimize(objective, n_trials=1)
    if os.path.exists("./STOP"):
        print("STOP file detected. Optymalizacja zatrzymana.")
        break

print("Step 19/22: Pobieranie najlepszych parametrów")
best_params = study.best_params
best_params.update({
    "device": "gpu",
    "objective": "regression",
    "metric": "rmse",
    "bagging_freq": 5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
})

print("Step 20/22: Trenowanie finalnego modelu z najlepszymi parametrami")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
test_predictions = []

print("Step 21/22: Ewaluacja wyników cross-validation")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed), 1):
    print(f"  Fold {fold}/5")
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
    )

    val_pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
    y_val_orig = np.expm1(y_val)

    fold_score = np.sqrt(mean_squared_log_error(y_val_orig, val_pred))
    cv_scores.append(fold_score)

    test_pred = np.expm1(model.predict(test_processed, num_iteration=model.best_iteration))
    test_predictions.append(test_pred)

print("Step 22/22: Generowanie pliku z przewidywaniami")
print(f"Average RMSLE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

final_predictions = np.mean(test_predictions, axis=0)
submission = pd.DataFrame({"id": test["id"], "Premium Amount": final_predictions})
submission.to_csv("./submission-runfile001.csv", index=False)
print("Plik z przewidywaniami został zapisany jako 'submission-runfile001.csv'")


