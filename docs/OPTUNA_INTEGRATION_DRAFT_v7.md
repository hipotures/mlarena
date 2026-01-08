# OPTUNA_INTEGRATION_DRAFT_v7.md
**Wersja:** v7 („FAST→FULL promotion workflow”)  
**Status:** draft do review  
**Zakres:** integracja Optuna z generatorem/preprocessingiem w MLArena (Kaggle-scale subset, AutoGluon thin w pętli)

---

## 0. Co doprecyzowano w v7 (względem v6)

Doprecyzowano operacyjny workflow, który już stosujesz:

- **Optuna pracuje na FAST** (thin AutoGluon, stałe seedy/splity, szybki budżet).
- **FULL uruchamiasz poza Optuną**: bierzesz top-N template’ów z FAST, robisz kopię i podmieniasz konfigurację modelową na FULL (LGBM+XGB+CatBoost, stacking/bagging, dłuższy time_limit).
- Następnie wypuszczasz kilka najlepszych na Kaggle.

To jest zgodne z posiadaną korelacją FAST↔FULL↔Kaggle i stabilizuje pętlę wyszukiwania preprocessingu.

---

## 1. Cel integracji

- Optuna dobiera warianty i parametry preprocessingu/FE.
- Runner uruchamia preprocessing + AutoGluon(thin) i zwraca `score_fast`.
- Trial-e mogą działać równolegle (np. 8–16 procesów, defaultowo 1), aby wykorzystać CPU mimo jednowątkowych kroków preprocessingu.
- Wyniki są porównywalne: stały seed, stałe splity/foldy, stała konfiguracja FAST.

---

## 2. Podejście bazowe: Super-pipeline ze switchami

Stała kolejność etapów (przykład):
1) sanity / drop columns  
2) outliers (opcjonalnie; bez drop na val/test)  
3) imputation (lub none)  
4) encoding (lub none)  
5) feature engineering (lub none)  
6) transformations (np. binning/ranks/pca; lub none)  
7) scaling (lub none)  
8) feature selection (lub none)

Każdy etap:
- `stage_X ∈ {"none", "variant_a", "variant_b", ...}`
- parametry szczegółowe tylko gdy `stage_X != "none"`.

---

## 3. Walidacja i „znikające wiersze” (outliers)

### 3.1. Stałe splity/foldy i seed
- Foldy/split ustalasz na surowych danych przed preprocessingiem.
- Seed jest stały (deterministyczny dobór subsetu/splitu i deterministyka w miarę możliwości).

### 3.2. Polityka: val/test bez drop wierszy
Jeśli etap „outliers”:
- w `fit()` (train fold) może usuwać wiersze / modyfikować sample_weight,
- w `transform()` na val/test **nie usuwa wierszy** (co najwyżej: clipping/winsorize/flag).

---

## 4. Model w pętli tuningu: AutoGluon thin + promocja do FULL

### 4.1. FAST (w pętli Optuny) — konfiguracja „zamrożona”
Żeby Optuna optymalizowała preprocessing, a nie zmienność AutoML, FAST ma stałe ustawienia:
- tylko LGBM/XGB (2–4 modele), bez CatBoost,
- brak bagging/stacking,
- stały budżet (presets/time_limit), stałe parametry treningu,
- stały seed i stałe foldy.

Wynik Optuny to `score_fast`.

### 4.2. FULL (poza Optuną) — niezależna weryfikacja i selekcja
Proces dzienny/iteracyjny:
1) wybierz top-N template’ów z FAST (np. N=20–50),
2) skopiuj template i podmień „model template” na FULL:
   - LGBM + XGB + CatBoost
   - stacking/bagging
   - time_limit ~ 1h (testowane 1/4/8h; praktycznie 1h wystarcza do selekcji)
3) uruchom FULL i wybierz top-K (np. K=5) do submissions.

Wynik FULL (`score_full`) nie trafia do Optuny jako sygnał treningowy (inne warunki/model), chyba że utworzysz osobne study „FULL”.

### 4.3. Metryki korelacji jako uzasadnienie workflow
Ponieważ masz empirycznie:
- FAST vs FULL ~ 0.92 (korelacja rang/score),
- FULL vs Kaggle ~ 0.94,

to FAST jest dobrym „proxy” do wyszukiwania preprocessingu, a FULL jest sensowną walidacją i selekcją finalnych submission.

---

## 5. Równoległość i zasoby CPU

### 5.1. Domyślnie: 1 trial = 1 proces
- izolacja danych (brak data pollution),
- dobre wykorzystanie CPU przy jednowątkowych krokach preprocessingu.

### 5.2. Kontrola oversubscription
Ponieważ boostery są wielowątkowe:
- ogranicz wątki per trial (np. 2–4),
- dobierz liczbę workerów tak, aby nie przekroczyć liczby rdzeni.

---

## 6. Storage Optuny dla wielu workerów

Przy >5 workerach:
- PostgreSQL/MySQL jako storage,
- unikać SQLite (ryzyko locków przy równoległych zapisach).

---

## 7. Timeout i bezpieczniki runtime

Rekomendacje:
- twardy timeout na trial FAST (np. 20–30 min),
- opcjonalnie timeout per-etap (np. RFE),
- hard cap na rozmiary (np. maks. liczba cech po FE).

Po timeout:
- trial fail/pruned,
- zapisać: etap, parametry, czas, liczba cech.

---

## 8. Kara za czas / ograniczenia czasowe (opcjonalnie)

Jeśli runtime ma znaczenie, rozważ:
- penalizację w celu: `score_fast - λ * log1p(total_sec)`,
- albo constraint: `total_sec <= limit`.

Niezależnie: zapisuj `total_sec`, `preprocess_sec`, `n_features_out`.

---

## 9. Unifikacja logiki: wspólny rdzeń Pipeline (CLI vs Tune vs FULL)

Wymóg:
- jedna implementacja `Chain/Pipeline` (sklejanie + wykonanie),
- adaptery:
  - Tune(FAST): load subset → Chain.run → AutoGluon thin → score_fast
  - Full rerun: load (większy zbiór / full config) → Chain.run → AutoGluon FULL → score_full
  - CLI/repro: Load → Chain.run → Save

---

## 10. Artefakty i reprodukcja

Per trial FAST:
- pipeline (materialized YAML/JSON),
- `trial.params`,
- `metrics.json` z: `score_fast`, `total_sec`, `preprocess_sec`, `n_features_out`, `seed`, `split_id`.

Dla FULL:
- powiązanie do FAST triala (ID),
- `score_full`, `time_limit`, konfiguracja FULL.

---

## 11. Dodawanie nowych transformacji

Na start: możliwie pełny zestaw transformacji i brak zmian w trakcie study.  
Jeśli dodasz transformację: nowe study + warm-start najlepszymi konfiguracjami.

---

## 12. PoC/testy

1) izolacja danych (data pollution)  
2) outliers: val/test bez drop  
3) reprodukcja Tune(FAST) vs CLI  
4) reprodukcja FULL rerun (zapis konfiguracji FULL)

---

## 13. Plan wdrożenia (MVP)

1) Super-pipeline + reguły `none/skip` (EDA gating).  
2) Wspólny rdzeń `Chain/Pipeline`.  
3) Multi-process workers + Postgres storage.  
4) Optuna na FAST (AutoGluon thin).  
5) Timeout + metryki czasu/rozmiaru cech.  
6) Promocja top-N do FULL i wybór top-K submissions.

---
