# Kaggle Scraper - Automatyczne pobieranie danych z Kaggle

## ⚠️ UWAGA: TY uruchamiasz Chrome, nie skrypt!

Skrypt **TYLKO** łączy się z istniejącym Chrome przez CDP.

## Setup

### 1. Zainstaluj Playwright

```bash
uv sync
playwright install chromium
```

### 2. TY uruchom Chrome z CDP (w osobnym terminalu)

```bash
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```

### 3. TY zaloguj się na Kaggle

W uruchomionym Chrome przejdź do https://www.kaggle.com i zaloguj się.

## Użycie

### Podstawowe

```bash
python tools/kaggle_scraper.py playground-series-s5e11
```

Skrypt tylko **czyta** dane przez CDP - nie uruchamia przeglądarki!

### Z własnymi opcjami

```bash
python tools/kaggle_scraper.py playground-series-s5e11 \
    --output-dir playground-series-s5e11/data/kaggle_scrapes \
    --cdp-url http://localhost:9222
```

## Output

Tworzy 3 pliki JSON:
- `kaggle_scrape_TIMESTAMP.json` - Pełne dane
- `leaderboard_TIMESTAMP.json` - Accessibility tree leaderboard
- `submissions_TIMESTAMP.json` - Accessibility tree submissions

## Workflow po submission

```bash
# 1. Submit do Kaggle (CLI)
cd playground-series-s5e11/submissions
kaggle competitions submit -c playground-series-s5e11 \
    -f submission-20231116235642.csv \
    -m "AutoGluon baseline"

# 2. Jeśli Chrome nie działa - TY go uruchom:
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

# 3. W Chrome otwórz leaderboard/submissions

# 4. Scrape dane (skrypt się tylko łączy, nie uruchamia Chrome!)
python tools/kaggle_scraper.py playground-series-s5e11

# 5. Przejrzyj JSON i znajdź public score

# 6. Zaktualizuj tracker
python tools/submissions_tracker.py --project playground-series-s5e11 \
    update 1 --public 0.85123
```
