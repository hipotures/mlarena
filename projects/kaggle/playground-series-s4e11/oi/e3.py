import pandas as pd
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

def print_error(e, limit=100):
    """Wyświetla skrócony komunikat o błędzie: pierwsze i ostatnie 100 znaków."""
    error_str = str(e)
    if len(error_str) > 2 * limit:
        print("Wystąpił błąd:", error_str[:limit], "...", error_str[-limit:])
    else:
        print("Wystąpił błąd:", error_str)  # Wyświetla pełny komunikat, jeśli jest krótszy niż 200 znaków
try:
    # Wczytywanie danych
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    # Przygotowanie danych treningowych
    X = train.drop(['Depression', 'id'], axis=1)
    y = train['Depression']

    # Identyfikacja i wypełnianie kolumn kategorycznych
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    X[cat_features] = X[cat_features].fillna(X[cat_features].mode().iloc[0])
    X.fillna(X.mean(), inplace=True)

    # Definiowanie modeli
    model1 = CatBoostClassifier(iterations=10, cat_features=cat_features, verbose=0)
    model2 = XGBClassifier(n_estimators=10)
    model3 = LGBMClassifier(n_estimators=10)

    # Ewaluacja modeli za pomocą cross-validation
    scores1 = cross_val_score(model1, X, y, cv=5, scoring='accuracy')
    scores2 = cross_val_score(model2, X, y, cv=5, scoring='accuracy')
    scores3 = cross_val_score(model3, X, y, cv=5, scoring='accuracy')

    # Wyświetlanie wyników dokładności
    print(f'CatBoost Accuracy: {scores1.mean()}')
    print(f'XGBoost Accuracy: {scores2.mean()}')
    print(f'LGBM Accuracy: {scores3.mean()}')

    # Trenowanie finalnego modelu CatBoost
    final_model = CatBoostClassifier(iterations=10, cat_features=cat_features, verbose=0)
    final_model.fit(X, y)

    # Przygotowanie danych testowych
    test[cat_features] = test[cat_features].fillna(test[cat_features].mode().iloc[0])
    test.fillna(test.mean(), inplace=True)

    # Przewidywanie i zapis wyników do pliku
    predictions = final_model.predict(test.drop('id', axis=1))
    submission = pd.DataFrame({'id': test['id'], 'Depression': predictions})
    submission.to_csv('./working/submission.csv', index=False)

except Exception as e:
    print_error(e)  # Wywołanie funkcji do wyświetlenia zwięzłego błędu

