import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import optuna
from itertools import combinations

# Step 1/22: Importowanie bibliotek
print("Step 1/22: Importowanie bibliotek")

# Step 2/22: Wczytywanie danych treningowych i testowych
print("Step 2/22: Wczytywanie danych treningowych i testowych")
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# Step 3/22: Losowanie 10% danych treningowych i resetowanie indeksu
print("Step 3/22: Losowanie 10% danych treningowych i resetowanie indeksu")
train = train.sample(frac=0.1, random_state=43).reset_index(drop=True)

# Step 4/22: Oddzielanie cech od celu
print("Step 4/22: Oddzielanie cech od celu")
X = train.drop(["Premium Amount", "id"], axis=1)
y = np.log1p(train["Premium Amount"])
test_features = test.drop("id", axis=1)

# Step 5/22: Identyfikacja kolumn kategorycznych i numerycznych
print("Step 5/22: Identyfikacja kolumn kategorycznych i numerycznych")
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Step 6/22: Inicjalizacja imputerów dla brakujących wartości
print("Step 6/22: Inicjalizacja imputerów dla brakujących wartości")
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Step 7/22: Imputacja brakujących wartości w danych treningowych i testowych
print("Step 7/22: Imputacja brakujących wartości w danych treningowych i testowych")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test_features[num_cols] = num_imputer.transform(test_features[num_cols])
test_features[cat_cols] = cat_imputer.transform(test_features[cat_cols])

# Step 8/22: Tworzenie cech agregacyjnych, w tym EWMA
print("Step 8/22: Tworzenie cech agregacyjnych, w tym EWMA")
agg_features = []
agg_feature_names = []
mean_features = []
decay_factors = [0.3, 0.5, 0.7]

for cat_col in cat_cols:
    for num_col in ["Age", "Annual Income", "Health Score"]:
        # Basic aggregations
        agg_stats = (
            X.groupby(cat_col)[num_col].agg(["mean", "std", "min", "max"]).reset_index()
        )
        agg_stats.columns = [cat_col] + [
            f"{cat_col}_{num_col}_{stat}" for stat in ["mean", "std", "min", "max"]
        ]
        X = X.merge(agg_stats, on=cat_col, how="left")
        test_features = test_features.merge(agg_stats, on=cat_col, how="left")

        # EWMA calculations
        for alpha in decay_factors:
            # Sort by categorical variable to simulate temporal order
            temp_train = X.sort_values(by=[cat_col, num_col])
            temp_test = test_features.sort_values(by=[cat_col, num_col])

            # Calculate EWMA for each category
            ewma_dict = {}
            for category in temp_train[cat_col].unique():
                cat_values = temp_train[temp_train[cat_col] == category][num_col]
                ewma = pd.Series(cat_values).ewm(alpha=alpha).mean().values
                ewma_dict[category] = ewma

            # Add EWMA features
            ewma_name = f"{cat_col}_{num_col}_ewma_{alpha}"
            X[ewma_name] = X.apply(
                lambda row: (
                    ewma_dict[row[cat_col]][0] if row[cat_col] in ewma_dict else np.nan
                ),
                axis=1,
            )
            test_features[ewma_name] = test_features[cat_col].map(
                lambda x: ewma_dict[x][0] if x in ewma_dict else np.nan
            )
            agg_feature_names.append(ewma_name)

        mean_feature_name = f"{cat_col}_{num_col}_mean"
        mean_features.append(mean_feature_name)
        agg_feature_names.extend(
            [f"{cat_col}_{num_col}_{stat}" for stat in ["mean", "std", "min", "max"]]
        )

# Step 9/22: Tworzenie cech interakcyjnych między średnimi agregacji
print("Step 9/22: Tworzenie cech interakcyjnych między średnimi agregacji")
for feat1, feat2 in combinations(mean_features, 2):
    interaction_name = f"interaction_{feat1}_{feat2}"
    X[interaction_name] = X[feat1] * X[feat2]
    test_features[interaction_name] = test_features[feat1] * test_features[feat2]
    agg_feature_names.append(interaction_name)

# Step 10/22: Tworzenie cech binowanych dla ważnych kolumn numerycznych
print("Step 10/22: Tworzenie cech binowanych dla ważnych kolumn numerycznych")
important_num_cols = ["Age", "Annual Income", "Health Score"]
binner = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile")
X_binned = binner.fit_transform(X[important_num_cols])
test_binned = binner.transform(test_features[important_num_cols])

binned_feature_names = [
    f"{col}_bin_{i}" for col in important_num_cols for i in range(10)
]
X_binned_df = pd.DataFrame(X_binned, columns=binned_feature_names)
test_binned_df = pd.DataFrame(test_binned, columns=binned_feature_names)

# Step 11/22: Tworzenie cech polinomialnych
print("Step 11/22: Tworzenie cech polinomialnych")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[important_num_cols])
test_poly = poly.transform(test_features[important_num_cols])
poly_feature_names = [f"poly_{i}" for i in range(X_poly.shape[1])]

# Step 12/22: Target Encoding dla cech kategorycznych
print("Step 12/22: Target Encoding dla cech kategorycznych")
te = TargetEncoder()
X_cat_encoded = te.fit_transform(X[cat_cols], y)
test_cat_encoded = te.transform(test_features[cat_cols])

# Step 13/22: Skalowanie cech numerycznych
print("Step 13/22: Skalowanie cech numerycznych")
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols)
test_num_scaled = pd.DataFrame(
    scaler.transform(test_features[num_cols]), columns=num_cols
)

# Step 14/22: Dodawanie cech polinomialnych zaczynając od nieliniowych terminów
print("Step 14/22: Dodawanie cech polinomialnych zaczynając od nieliniowych terminów")
X_poly_df = pd.DataFrame(
    X_poly[:, len(important_num_cols) :],
    columns=poly_feature_names[len(important_num_cols) :],
)
test_poly_df = pd.DataFrame(
    test_poly[:, len(important_num_cols) :],
    columns=poly_feature_names[len(important_num_cols) :],
)

# Step 15/22: Skalowanie cech agregacyjnych
print("Step 15/22: Skalowanie cech agregacyjnych")
X_agg = X[agg_feature_names]
test_agg = test_features[agg_feature_names]
X_agg_scaled = pd.DataFrame(scaler.fit_transform(X_agg), columns=agg_feature_names)
test_agg_scaled = pd.DataFrame(scaler.transform(test_agg), columns=agg_feature_names)

# Step 16/22: Łączenie wszystkich cech
print("Step 16/22: Łączenie wszystkich cech")
X_processed = pd.concat(
    [X_num_scaled, X_cat_encoded, X_poly_df, X_binned_df, X_agg_scaled], axis=1
)
test_processed = pd.concat(
    [test_num_scaled, test_cat_encoded, test_poly_df, test_binned_df, test_agg_scaled],
    axis=1,
)

# Step 17/22: Definicja funkcji celu dla Optuna
print("Step 17/22: Definicja funkcji celu dla Optuna")
def objective(trial):
    params = {
        "device" : "gpu",
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
        )

        val_pred = np.expm1(model.predict(X_val))
        y_val_orig = np.expm1(y_val)

        fold_score = np.sqrt(mean_squared_log_error(y_val_orig, val_pred))
        cv_scores.append(fold_score)

    return np.mean(cv_scores)

# Step 18/22: Optymalizacja hyperparametrów za pomocą Optuna
print("Step 18/22: Optymalizacja hyperparametrów za pomocą Optuna")
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# Step 19/22: Pobieranie najlepszych parametrów
print("Step 19/22: Pobieranie najlepszych parametrów")
best_params = study.best_params
best_params.update(
    {
        "device" : "gpu",
        "objective": "regression",
        "metric": "rmse",
        "bagging_freq": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
)

# Step 20/22: Trenowanie finalnego modelu z najlepszymi parametrami
print("Step 20/22: Trenowanie finalnego modelu z najlepszymi parametrami")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
test_predictions = []

# Step 21/22: Ewaluacja wyników cross-validation
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

    val_pred = np.expm1(model.predict(X_val))
    y_val_orig = np.expm1(y_val)

    fold_score = np.sqrt(mean_squared_log_error(y_val_orig, val_pred))
    cv_scores.append(fold_score)

    test_pred = np.expm1(model.predict(test_processed))
    test_predictions.append(test_pred)

# Step 22/22: Generowanie pliku z przewidywaniami
print("Step 22/22: Generowanie pliku z przewidywaniami")
print(f"Average RMSLE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

final_predictions = np.mean(test_predictions, axis=0)
submission = pd.DataFrame({"id": test["id"], "Premium Amount": final_predictions})
submission.to_csv("./submission-runfile001.csv", index=False)
print("Plik z przewidywaniami został zapisany jako 'submission-runfile001.csv'")

