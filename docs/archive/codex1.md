# Uwagi Codex do dokumentu `docs/ml_code_separation_design.md`

1. **Kontrakt konfiguracji** – obecny słownik `config` łączy ścieżki systemowe, parametry Autogluon i flagi preprocessingowe. Warto ustalić hierarchię nadpisywania (`get_default_config()` → config projektu → wpis w `model.yaml`/`preprocess.yaml` → CLI) oraz rozdzielić przestrzeń nazw (np. `config['system']`, `config['dataset']`, `config['hyperparameters']`) albo zastosować lekką dataclass, żeby IDE/mypy mogły łatwiej wykrywać brakujące pola (`docs/ml_code_separation_design.md:58-156`).

2. **Preprocessing ze stanem** – runner wywołuje `preprocess()` oddzielnie dla train/val/test, więc dopasowany scaler/encoder nie ma jak dostać się do inferencji (`docs/ml_code_separation_design.md:188-196`). Możliwe rozszerzenia:  
   - `train()` zwraca `(model, artifacts)` i runner przekazuje `artifacts` do `predict()`;  
   - albo opcjonalna funkcja `prepare_artifacts()` wołana przed treningiem, która zwraca obiekty do późniejszego użycia.

3. **Niestandardowe ładowanie danych** – `_load_data()` w MLRunnerze jest globalne (`docs/ml_code_separation_design.md:186-190`). Modele wymagające customowego łączenia plików mogłyby eksportować `load_data(config)`; runner sprawdzi `hasattr` i pozwoli zastąpić domyślny loader.

4. **Wsparcie dla ensemble/pipeline** – template `ensemble` zakłada dostęp do predykcji innych modeli (`docs/ml_code_separation_design.md:151-155`), ale pętla `run()` obsługuje tylko pojedynczy model (`docs/ml_code_separation_design.md:197-205`). Potrzebny plan na wykonywanie wielu szablonów sekwencyjnie (np. deklaracja zależności w YAML i buforowanie predykcji out-of-fold).

5. **Rola `code/preprocessing/`** – katalog wymieniony w strukturze (`docs/ml_code_separation_design.md:18-27`), ale runner go nie używa. Dobrze opisać, czy to repozytorium wspólnych helperów importowanych przez modele, czy planujesz automatyczne ładowanie modułów stamtąd.

6. **Przyszła ergonomia** – dynamiczne importy (`docs/ml_code_separation_design.md:171-179`) dają elastyczność, lecz ograniczają narzędzia statyczne. Rozważ wygenerowanie stubów (np. `__all__` lub moduł rejestru), żeby IDE mogło ogarnąć dostępne modele.
