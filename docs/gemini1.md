- Ogólny kierunek: Separacja „czystego” ML w code/models/*.py oraz warstwy infrastruktury w runnerze dobrze rozwiązuje opisane
    problemy czytelności; programista skupia się na train/predict i opcjonalnym preprocess, a resztę załatwia MLRunner (docs/
    ml_code_separation_design.md:5-146, 160-231).
  - Kontrakt na modele: Warto doprecyzować kontrakt, np. przez mały interfejs/klasę bazową albo dataclass konfiguracyjny
    (łatwiejsza walidacja i autouzupełnianie). Aktualne podejście z dowolnym słownikiem (config) daje swobodę, ale utrudnia
    wykrywanie brakujących pól i pisanie narzędzi statycznych (docs/ml_code_separation_design.md:33-121).
  - Preprocessing i stan: Ponieważ preprocess wywoływany jest niezależnie na train/val/test, wszelki stan (np. fitted scaler)
    musiałby być wyuczony w train i odtworzony ręcznie przed predict. Można rozważyć przekazywanie kontekstu pozwalającego
    cache’ować dopasowane obiekty albo wprowadzić hook prepare_artifacts() zwracający obiekt z transformacjami (docs/
    ml_code_separation_design.md:87-111, 173-207).
  - Obsługa predykcji: Przyjęcie, że wszystkie modele zwrócą DataFrame z kolumną id i target, działa dla standardowych konkursów,
    ale warto przewidzieć projekty wielokolumnowe (np. wielowyjściowe regresje, segmentacje). Runner mógłby pobierać strukturę z
    config lub sample_submission zamiast wymuszać konkretny kształt (docs/ml_code_separation_design.md:99-117).
  - Template’y i konfigi: Fajnie, że templates/model.yaml mapuje nazwę template → model + parametry, ale przyda się jasna
    kolejność nakładania: baza z code/models/.../get_default_config(), globalny config projektu, template, a na koniec flagi CLI.
    Dobrze też zautomatyzować walidację YAML (np. schema), żeby literówki w model lub brak kluczy wychwycić przed uruchomieniem
    (docs/ml_code_separation_design.md:123-158).
  - Migracja/równoległość: Planowane współistnienie starego runnera i nowej architektury wydaje się sensowne; rozważyłbym
    jednak szybkie narzędzia do „wrapowania” istniejących skryptów (np. generator pliku w code/models na bazie aktualnego
    autogluon_runner.py), by zminimalizować ręczne przenoszenie (docs/ml_code_separation_design.md:233-276).
