TODO:
1. Masz już infrastrukturę, w której preprocessing składa się z kilku kroków (sub-modułów) łączonych w łańcuch przez template’y `preprocess`, a każdy krok implementuje wspólne API (`fit_transform` / `transform`) i jest konfigurowany przez YAML.
2. Chcesz zdefiniować rodzinę **uniwersalnych modułów preprocesingu** (np. imputacja braków, różne strategie selekcji cech, obróbka kategorii, detekcja/obsługa dryfu itp.), które można dowolnie zestawiać w template’ach w różnej kolejności.
3. Każdy moduł ma mieć **jeden wspólny kod**, ale różne zachowania wybierane przez parametry w template (np. `method: "variance_threshold" | "select_k_best"` oraz parametry typu `k`, procent cech do pozostawienia, próg wariancji, itp.).
4. Oczekujesz ode mnie **listy takich sub-modułów**, bez kodu, gdzie dla każdego w max. 5–10 zdaniach opiszę:

   * co dokładnie robi krok,
   * jakich bibliotek/narzędzi używa (np. `sklearn`, `pandas`, ewentualnie `category_encoders` itp.),
   * jakie parametry konfiguracyjne ma mieć w template (np. wybór algorytmu, progi, liczby cech, tryby działania).
5. Następnie samodzielnie złożysz różne template’y (różne kombinacje kroków) i odpalisz modelowanie (głównie AutoGluon, w tym wariant z natywnymi kategoriami), żeby empirycznie sprawdzić, które kombinacje preprocessing + model dają najlepszy wynik.


LIST TODO:
Poniżej proponowana lista głównych sub-modułów preprocessingowych pod Kaggle (tablice, AutoGluon), projektowanych tak, żeby można je było składać w dowolne łańcuchy i sterować samą konfiguracją w template’ach.

---
Globalne Template'y (config/templates/preprocess.yaml)

  - ✅ Dodane 12 placeholder'ów dla sub-modułów:
    a. sanity_check - Typy, czyszczenie, duplikaty
    b. imputer - Imputacja wartości brakujących
    c. rare_category_handler - Obsługa rzadkich kategorii
    d. encoder - Kodowanie kategorii
    e. scaler - Skalowanie i transformacje numeryczne
    f. feature_selector - Selekcja cech
    g. drift_detector - Detekcja dryfu train-test
    h. feature_engineer - Feature engineering
    i. target_transformer - Transformacje targetu (regresja)
    j. imbalance_handler - Obsługa niezbalansowanych klas
    k. outlier_handler - Obsługa outlierów
    l. autogluon_booster - Optymalizacja pod AutoGluon
  - ✅ Meta-template'y (chains):
    - minimal_pipeline
    - standard_pipeline
    - full_universal_pipeline


### 1. Moduł sanity check / typy / podstawowe czyszczenie

* Cel: ujednolicenie typów i szybkie odfiltrowanie oczywistych problemów zanim ruszy reszta pipeline’u.
* Narzędzia: `pandas`, `numpy`, ewentualnie lekkie funkcje z `sklearn` (np. `SimpleImputer` tylko dla bardzo prostych przypadków).
* Funkcje: wymuszenie typów kolumn (numeryczne, kategoryczne, daty, tekst), wykrywanie „prawie stałych” kolumn, kolumn pustych, duplikatów wierszy/ID, nielogicznych wartości (np. inf, -inf, NaN w kolumnach, które nie powinny ich mieć).
* Parametry w template: słownik mapujący kolumny na docelowe typy (`column_types_override`), progi dla usuwania kolumn (`min_unique_fraction`, `max_missing_fraction`), flaga usuwania duplikatów (`drop_duplicates: true/false`), lista kolumn do zignorowania w dalszych krokach (`ignore_columns`).
* Efekt: generuje raport (log + artefakt CSV/JSON) z wykrytymi problemami oraz wersję danych oczyszczonych z najbardziej oczywistych błędów.

---

### 2. Moduł imputacji wartości brakujących

* Cel: uniwersalna imputacja braków, z możliwością wyboru strategii osobno dla typów/kolumn i sterowana konfiguracją.
* Narzędzia: `sklearn.impute` (`SimpleImputer`, `KNNImputer`, `IterativeImputer`), `pandas`.
* Funkcje: różne strategie dla kolumn numerycznych (`mean`, `median`, `most_frequent`, `constant`, `knn`, `iterative`) i kategorycznych (`most_frequent`, `constant`, „nowa kategoria”), opcjonalne traktowanie outlierów jako braków przed imputacją.
* Parametry w template: globalna strategia dla typów (`numeric_strategy`, `categorical_strategy`), słownik nadpisania strategii dla wybranych kolumn (`column_strategies`), wartość zastępcza dla `constant` (`fill_value`), parametry KNN (`knn_n_neighbors`) i modeli w `IterativeImputer` (`iterative_estimator`, `max_iter`), flaga „treat_outliers_as_na”.
* Efekt: zapisuje imputowane wersje danych oraz artefakt z maskami braków (np. liczba/odsetek imputowanych wartości per kolumna) dla późniejszej analizy.

---

### 3. Moduł obsługi rzadkich i wysokokardynalnych kategorii

* Cel: redukcja wysokiej krotności kategorii i rzadkich wartości przed właściwym kodowaniem, żeby zmniejszyć overfitting i rozmiar feature space.
* Narzędzia: `pandas`, `numpy`.
* Funkcje: grupowanie rzadkich kategorii do etykiety typu `"__RARE__"`, obcinanie do top-K najczęstszych kategorii, detekcja potencjalnych ID (wysoka unikalność) i opcjonalne oznaczenie takiej kolumny jako „do usunięcia” lub „tylko dla liczników/statystyk”.
* Parametry w template: minimalna częstość (`min_freq`), minimalny udział (`min_freq_ratio`), maksymalna liczba kategorii (`top_k`), nazwa kategorii zbiorczej (`rare_label`), flaga detekcji ID (`detect_id_like_columns`) i próg unikalności (`id_unique_fraction_threshold`), lista kolumn, które nie powinny być ruszane (`protected_categorical_columns`).
* Efekt: zredukowane kategorie, mniejsza liczba poziomów w kolejnych encoderach, artefakt z mapowaniem „stara_kategoria → nowa_kategoria/RARE”.

---

### 4. Moduł kodowania zmiennych kategorycznych

* Cel: uniwersalne kodowanie kategorii z wyborem algorytmu i parametrów per template, kompatybilne z trybem „brak kodowania” dla AutoGluon (natywne kategorie).
* Narzędzia: `category_encoders`, `sklearn.preprocessing` (`OneHotEncoder`, `OrdinalEncoder`), `pandas`.
* Funkcje: strategie kodowania (`none` – zostaw surowe kategorie, `one_hot`, `ordinal`, `target_mean`, `catboost`, `hashing`), opcjonalne utrzymywanie zarówno oryginalnej kolumny jak i zakodowanej wersji, kontrola nad obsługą nieznanych kategorii i drop-first.
* Parametry w template: `encoding_method` (np. `none|one_hot|ordinal|target_mean|catboost|hashing`), lista kolumn do kodowania i/lub wykluczenia (`include_cols`, `exclude_cols`), `drop_first`, `handle_unknown` (`ignore|use_encoded_value` z `unknown_value`), rozmiar przestrzeni dla hashowania (`hash_dim`), parametry target encodingu (liczba foldów, smoothing, random_state, scheme leakage-safe z K-fold).
* Efekt: zestaw kolumn numerycznych gotowych do modelu oraz artefakty z fitted encoderami, które można odtworzyć przy predykcji.

---

### 5. Moduł skalowania i transformacji zmiennych numerycznych

* Cel: standaryzacja i transformacje rozkładów dla modeli wrażliwych na skalę/rozkład oraz test różnych wariantów w pipeline’ach.
* Narzędzia: `sklearn.preprocessing` (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `QuantileTransformer`), `numpy`.
* Funkcje: wybór metody skalowania (brak, standard, min-max, robust, quantile-normal, quantile-uniform), opcjonalne log-transformacje (log1p, log10) i winsoryzacja/clip wdł. kwantyli.
* Parametry w template: `scaling_method` (`none|standard|minmax|robust|quantile_normal|quantile_uniform`), lista kolumn do skalowania (`numeric_include`, `numeric_exclude`), `log_transform` (bool lub lista kolumn), parametry winsoryzacji (`clip_lower_quantile`, `clip_upper_quantile`), parametry `QuantileTransformer` (liczba kwantyli, random_state).
* Efekt: liczby na spójnych skalach, potencjalnie bliższych normalności, które można łatwo porównywać między różnymi setupami.

---

### 6. Moduł selekcji cech (feature selection)

* Cel: systematyczne zmniejszanie wymiaru cech różnymi metodami, sterowane konfiguracją, z jednym kodem obsługującym wiele algorytmów.
* Narzędzia: `sklearn.feature_selection` (`VarianceThreshold`, `SelectKBest`, `mutual_info_classif/regression`, `RFE`), modele bazowe (`sklearn.ensemble.RandomForest*`, `lightgbm`, `xgboost` – jeśli dostępne), `sklearn.linear_model` (L1).
* Funkcje: kilka trybów: prosta filtracja (wariancja, mutual information, korelacja z targetem), embedded (L1, feature importances z modelu drzewiastego), wrapper (RFE), a także tryb „top_N_percent” zamiast sztywnego K.
* Parametry w template: `selection_method` (`variance|mi|correlation|model_importance|l1|rfe|none`), liczba cech (`k_features`) lub udział (`keep_fraction`), progi (np. `min_variance`, `min_importance`), typ modelu dla importance (`importance_model_type: lgbm|xgb|rf` + parametry jak `n_estimators`, `max_depth`), random_state, maksymalna frakcja cech do wyrzucenia w jednym kroku (`max_drop_fraction`).
* Efekt: zredukowany zestaw cech plus artefakt z rankingiem/importance każdej cechy, łatwy do analizy między eksperymentami.

---

### 7. Moduł detekcji i filtrowania dryfu / różnic train–test

* Cel: wykrywanie cech o silnie różnym rozkładzie między train i test oraz opcjonalne ich usuwanie lub oznaczanie, co jest bardzo ważne w wielu konkursach Kaggle.
* Narzędzia: `scipy.stats` (testy KS/chi-kwadrat), proste modele klasyfikacyjne do detekcji dryfu (`lightgbm`, `sklearn.ensemble.RandomForestClassifier`), obliczanie PSI (Population Stability Index).
* Funkcje: liczenie miar różnicy rozkładów (PSI, KS dla ciągłych, chi-kwadrat dla kategorycznych, AUC modelu „train vs test” dla każdej cechy), flagowanie problematycznych cech oraz opcjonalne ich usuwanie lub tylko logowanie.
* Parametry w template: `drift_metric` (`psi|ks|chi2|model_auc`), progi (`max_psi`, `max_ks`, `max_pvalue`, `min_auc`), strategia (`action: none|drop|flag_only`), `max_drop_fraction` (żeby nie wyciąć połowy featurów jednym testem), lista cech wyłączonych z detekcji (`exclude_cols`).
* Efekt: czystszy zestaw cech bardziej stabilnych między train i test oraz raport z miarami dryfu per cecha.

---

### 8. Moduł generowania cech interakcyjnych i agregacji (feature engineering)

* Cel: systematyczne tworzenie nowych cech interakcyjnych i agregacyjnych w kontrolowany sposób zamiast ad-hoc ręcznego grzebania.
* Narzędzia: `pandas` (groupby/agg), `sklearn.preprocessing.PolynomialFeatures` dla interakcji numerycznych, ewentualnie własne proste operacje (ratio, różnice).
* Funkcje: tworzenie interakcji numerycznych (suma, różnica, iloraz, produkt), wielomianów do zadanego stopnia, agregacji grupowych (średnie, odchylenia, liczniki) po zadanych kluczach (np. użytkownik, produkt, czas), opcjonalnie z kontrolą leakage (agregacje liczone na train z K-fold, a test z pełnego train).
* Parametry w template: listy kolumn do interakcji (`numeric_pairs`, `group_keys`, `group_value_cols`), typy interakcji (`interaction_types: add|sub|mul|div`), stopień wielomianów (`poly_degree`), agregacje (`aggs: mean|std|min|max|count|nunique`), flaga fold-aware (`leakage_safe_aggs: true/false`) z parametrami podziału (`n_splits`, `random_state`).
* Efekt: zestaw dodatkowych, potencjalnie silnych featurów, których liczba jest kontrolowana przez parametry template, plus artefakt z opisem, które nowe kolumny powstały z jakiej kombinacji.

---

### 9. Moduł transformacji targetu (dla regresji)

* Cel: umożliwienie testowania różnych transformacji zmiennej celu, co często poprawia jakość modeli regresyjnych i stabilizuje rozkłady błędów.
* Narzędzia: `numpy`, `scipy.stats` (`boxcox`, `yeo_johnson`), `sklearn.preprocessing.PowerTransformer`.
* Funkcje: transformacje typu `log1p`, `BoxCox`, `Yeo–Johnson`, opcjonalne clipowanie skrajnych wartości przed transformacją i przechowywanie parametrów transformacji do odwrócenia predykcji.
* Parametry w template: `target_transform` (`none|log1p|boxcox|yeo_johnson`), progi clipowania (`clip_lower_quantile`, `clip_upper_quantile`), sposób obchodzenia się z wartościami ≤ 0 dla log/BoxCox (np. `shift_before_log` i wartość przesunięcia), parametry PowerTransformer (`standardize: true/false`).
* Efekt: model trenuje na przetransformowanym targetcie, a moduł dostarcza funkcje/artefakty do odwrotnej transformacji predykcji na etapie `predict`.

---

### 10. Moduł obsługi niezbalansowanych klas (classification imbalance handler)

* Cel: umożliwienie systematycznego testowania strategii radzenia sobie z niezbalansowanymi klasami przed/na etapie trenowania modeli.
* Narzędzia: `imbalanced-learn` (`RandomUnderSampler`, `RandomOverSampler`, `SMOTE`, `SMOTENC`, `ADASYN`), `pandas`.
* Funkcje: różne strategie resamplingu: brak, oversampling, undersampling, SMOTE/ADASYN, w trybie globalnym lub per-fold (jeśli ściśle powiązane z CV), opcjonalne generowanie sample-weights zamiast fizycznej zmiany rozkładu.
* Parametry w template: `imbalance_method` (`none|class_weight|random_over|random_under|smote|smotenc|adasyn`), `sampling_strategy` (np. docelowy stosunek minor/major), `use_sample_weights: true/false`, lista kolumn kategorycznych dla `SMOTENC`, random_state.
* Efekt: zbalansowane dane do trenowania lub wektory wag przekazywane do modelu, co można łatwo porównać między eksperymentami.

---

### 11. Moduł obsługi outlierów

* Cel: standaryzowany sposób wykrywania i obchodzenia się z outlierami, który można łatwo porównywać między template’ami.
* Narzędzia: `numpy`, `pandas`, `scipy.stats` (z-score), ewentualnie `sklearn.ensemble.IsolationForest` dla bardziej zaawansowanych metod.
* Funkcje: proste metody oparte na kwantylach (clipowanie), IQR (outlier = poza [Q1 − k*IQR, Q3 + k*IQR]), z-score, oraz opcjonalnie IsolationForest; outliery mogą być clipowane, zamieniane na NaN (do późniejszej imputacji) lub oznaczane dodatkowymi flagami binarnymi.
* Parametry w template: `outlier_method` (`none|quantile|iqr|zscore|isolation_forest`), parametry per metoda (`lower_quantile`, `upper_quantile`, `iqr_factor`, `zscore_threshold`, parametry modelu IF), `action` (`clip|set_na|flag_only`), lista kolumn do analizy (`include_cols`, `exclude_cols`).
* Efekt: bardziej stabilne numeryczne cechy, mniej ekstremalnych wartości niszczących modele, plus raport z odsetkiem outlierów per kolumna.

---

### 12. Moduł „AutoGluon booster” pod kategorie i typy

* Cel: specjalizowany moduł pod AutoGluon, który przygotowuje typy i dodatkowe cechy tak, żeby AutoGluon mógł maksymalnie wykorzystać swoje wbudowane możliwości (w tym kategorie).
* Narzędzia: `pandas`, `numpy`, `category_encoders`, ewentualnie proste agregacje/feature engineering dopasowane do AutoGluon, ale bez psucia jego natywnej obsługi typów.
* Funkcje: poprawne rzutowanie typów (daty → datetime, flagi → bool, tekst → object/string), opcjonalne tworzenie kopii wybranych kategorii w postaci liczności/frequency encodingu (zostawiając oryginał), prostych count-encodingów (np. liczba wystąpień wartości w train), oznaczanie kolumn tekstowych do „text features” AutoGluon.
* Parametry w template: `autogluon_mode` (`raw_categories|boosted_categories`), listy kolumn do dodatkowych liczników/frequency (`freq_encoding_cols`), parametry tych encodingów (np. minimalna liczba wystąpień, czy normalizować przez liczebność datasetu), flaga automatycznej detekcji kolumn tekstowych (`auto_text_detection`), flaga wymuszania typów (`enforce_dtypes`).
* Efekt: dane idealnie przygotowane pod AutoGluon z możliwością testowania, czy dodatkowe encodery/liczniki pomagają w porównaniu z „gołym” trybem natywnych kategorii.

### 13. Moduł transformacji danych czasowych, dat i godzin

* Cel: ujednolicenie reprezentacji dat/czasu oraz generowanie bogatych cech czasowych przy zachowaniu kontroli nad ilością featurów i ryzykiem leakage.
* Narzędzia: `pandas` (`to_datetime`, atrybuty `.dt`), ewentualnie `dateutil`, proste własne funkcje do obliczania różnic czasu i cyklicznych transformacji (sin/cos) dla cech typu godzina/dzień tygodnia.
* Funkcje: rzutowanie kolumn na `datetime`, parsowanie wg zadanego formatu lub automatycznie, generowanie cech pochodnych (rok, miesiąc, dzień, godzina, minuta, dzień tygodnia, numer tygodnia, kwartał, pora dnia, weekend/święto – jeśli dostarczysz kalendarz), obliczanie różnic czasowych między parami kolumn (np. „czas_od_rejestracji”), opcjonalne cykliczne kodowanie cech okresowych (godzina, dzień tygodnia, miesiąc).
* Parametry w template: lista kolumn do parsowania jako daty (`datetime_cols`) z opcjonalnymi formatami (`datetime_formats`), lista kolumn do generowania cech pochodnych (`expand_datetime_cols`), wybór zestawu cech (`time_features_set: basic|extended|custom`), definicje różnic czasowych (`time_diff_pairs: [[col_start, col_end, new_name], ...]`), konfiguracja cyklicznych transformacji (`cyclical_features: [hour, dayofweek, month]`), flaga usuwania oryginalnej kolumny datetime po ekspansji (`drop_original_datetime`).
* Efekt: zestaw dobrze opisanych cech czasowych (z kontrolowaną granularnością), które można łatwo włączać/wyłączać i porównywać między eksperymentami, oraz spójne typy datetime dla całego pipeline’u.


---
Każdy zakończony moduł (kod+dokumentacja) ma się sfinalizować commitem w git


Na tej bazie możesz budować template’y typu:

* minimalistyczny (`sanity_check → imputacja → AutoGluon booster`),
* feature-heavy (`sanity_check → imputacja → rare_category → encoder → interactions → drift_filter → feature_selector → AutoGluon booster`),
* stabilność/dryf-oriented itd., a wszystko sterowane wyłącznie parametrami w YAML.

