# Scripts

## subset_compat.py

Skrypt ocenia, czy mniejszy podzbior treningowy jest "kompatybilny" z pełnym
zbiorem (drift, PSI, adversarial AUC). Przydaje się do decyzji o treningu na
subsetach w trybie szybkim.

Przykład:

```bash
uv run scripts/subset_compat.py \
  --train-path projects/kaggle/playground-series-s6e1/data/train.csv \
  --eda-json projects/kaggle/playground-series-s6e1/experiments/eda/state.json \
  --config-py projects/kaggle/playground-series-s6e1/code/utils/config.py \
  --out-json subset_report.json
```

Wyjście:
- Tabela z metrykami dla kolejnych frakcji (np. 1.0, 0.9, 0.8...).
- Opcjonalny JSON z wynikami i progami (`--out-json`).

Uwagi:
- Stratified sampling wymaga, by etykiety do stratyfikacji miały długość
  całego zbioru, a rozmiar próbki był podawany osobno.
- Jeżeli `target` ma NaN, stratyfikacja jest wyłączona.
