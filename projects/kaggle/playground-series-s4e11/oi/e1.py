import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    # Wczytywanie danych
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    # Przygotowanie danych
    X = train.drop(['Depression', 'id'], axis=1)
    y = train['Depression']

    # Identyfikacja kolumn kategorycznych
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    X.fillna(X.mean(), inplace=True)
    test.fillna(test.mean(), inplace=True)

    # Tworzenie i trenowanie modeli
    model1 = CatBoostClassifier(iterations=10, cat_features=cat_features, verbose=0)
    model2 = XGBClassifier(n_estimators=10)
    model3 = LGBMClassifier(n_estimators=10)

    # Ewaluacja modeli
    scores1 = cross_val_score(model1, X, y, cv=5, scoring='accuracy')
    scores2 = cross_val_score(model2, X, y, cv=5, scoring='accuracy')
    scores3 = cross_val_score(model3, X, y, cv=5, scoring='accuracy')

    print(f'CatBoost Accuracy: {scores1.mean()}')
    print(f'XGBoost Accuracy: {scores2.mean()}')
    print(f'LGBM Accuracy: {scores3.mean()}')

    # Trening finalnego modelu i generowanie predykcji
    final_model = CatBoostClassifier(iterations=10, cat_features=cat_features, verbose=0)
    final_model.fit(X, y)
    predictions = final_model.predict(test.drop('id', axis=1))
    
    # Przygotowanie pliku wyjściowego
    submission = pd.DataFrame({'id': test['id'], 'Depression': predictions})
    submission.to_csv('./working/submission.csv', index=False)

except Exception as e:
    print("Wystąpił błąd:", str(e))  # Wyświetlenie zwięzłego błędu
