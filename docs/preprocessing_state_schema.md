# Preprocessing Central State Schema (Unified Pipeline)

W trybie **Unified Pipeline** (In-Memory), całe przetwarzanie odbywa się w pamięci, a stan całego łańcucha jest monitorowany za pomocą jednego, centralnego pliku `state.json` umieszczonego w katalogu hasha eksperymentu.

## Lokalizacja pliku
`projects/kaggle/<slug>/experiments/<chain-id>/<hash>/state.json`

## Cykl życia i aktualizacja
1.  **Inicjalizacja**: Plik jest tworzony w momencie startu potoku z listą wszystkich kroków i statusem `running`.
2.  **Start kroku**: Przed uruchomieniem każdego modułu, wpis dla tego kroku jest aktualizowany o `started_at` i status `running`.
3.  **Koniec kroku**: Po zakończeniu modułu, wpis jest aktualizowany o `finished_at`, `duration`, `shapes` oraz `custom_module_state`.
4.  **Heartbeat**: Przy każdej zmianie aktualizowane jest pole `last_heartbeat`.
5.  **Finalizacja**: Po wykonaniu wszystkich kroków, główny status potoku zmienia się na `completed`.

## Struktura JSON

```json
{
  "experiment_id": "pre-chain-id/hash-id",
  "project": "project-name",
  "status": "running|completed|failed",
  "pipeline_progress": {
    "status": "running|completed|failed",
    "total_steps": 13,
    "current_step_idx": 5,
    "start_time": "2026-01-12T03:18:05.123Z",
    "end_time": null,
    "steps": [
      {
        "name": "0-sanity_check",
        "module": "sanity_check",
        "status": "completed",
        "started_at": "2026-01-12T03:18:05.818Z",
        "finished_at": "2026-01-12T03:18:06.046Z",
        "duration": 0.23,
        "shapes": {
          "train_before": [630000, 13],
          "train_after": [630000, 13],
          "test_before": [270000, 12],
          "test_after": [270000, 12]
        }
      },
      {
        "name": "1-feature_selector",
        "module": "feature_selector",
        "status": "running",
        "started_at": "2026-01-12T03:18:06.100Z",
        "finished_at": null,
        "duration": null
      }
    ]
  }
}
```

## Opis kluczowych pól

### Obiekt główny
- **`status`**: Ogólny stan eksperymentu.
- **`pipeline_progress`**: Kontener dla metadanych potoku In-Memory.
- **`total_steps`**: Całkowita liczba kroków zdefiniowana w szablonie.

### Obiekt kroku (w liście `steps`)
- **`name`**: Pełna nazwa kroku (np. `10-feature_selector`).
- **`duration`**: Czas trwania przetwarzania w sekundach (format `N.N`).
- **`shapes`**: Kształty wszystkich 5 wspieranych kontenerów danych (`train`, `test`, `val`, `eval`, `orig`). Przechowywane jako krotki `[rows, cols]`.
- **`custom_module_state`**: Miejsce na specyficzne dane wyjściowe modułu (np. `weights_path` dla AV).

## Monitorowanie
Zaleca się używanie pola `current_step_idx` oraz `last_heartbeat` do wizualizacji postępu w Dashboardach i wykrywania błędów bez konieczności zaglądania w logi tekstowe.
