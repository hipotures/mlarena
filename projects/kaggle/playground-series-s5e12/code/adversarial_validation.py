from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import argparse
import pandas as pd
from autogluon.tabular import TabularPredictor


AV_LABEL_COL = "__is_test__"
AV_PROB_COL = "av_prob"


def compute_adversarial_weights(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: Optional[str] = None,
    drop_columns: Optional[Iterable[str]] = None,
    presets: str = "medium_quality_faster_train",
    time_limit: int = 600,
    output_dir: str | Path = "autogluon_av_model",
) -> pd.DataFrame:
    """
    Trenuje classifier "train vs test" i zwraca dla train:
        id_column, av_prob

    av_prob = P(„ten rekord wygląda jak test”) z modelu adversarial.
    Możesz użyć tego jako wagi sample_weight albo do budowy test-like CV.

    Parametry
    ---------
    train_df, test_df : po tym samym preprocesingu (kolumny muszą się zgadzać poza targetem)
    id_column         : nazwa kolumny z identyfikatorem rekordu
    target_column     : nazwa kolumny targetu (jeśli jest w train_df – usuwamy z ficzerów AV)
    drop_columns      : dodatkowe kolumny do wyrzucenia z cech (np. id, target-leak)
    presets           : preset AutoGluon do AV
    time_limit        : limit czasu na fit (sekundy)
    output_dir        : gdzie zapisać model AutoGluon AV
    """

    drop_columns = list(drop_columns) if drop_columns is not None else []
    if target_column is not None:
        drop_columns = list(set(drop_columns) | {target_column})

    # Kopie, żeby nie nadpisywać wejścia
    train_adv = train_df.copy()
    test_adv = test_df.copy()

    # Etykieta domeny: 0 = train, 1 = test
    train_adv[AV_LABEL_COL] = 0
    test_adv[AV_LABEL_COL] = 1

    # Upewnij się, że kolumny są zgodne (poza targetem) – wspólne cechy
    common_cols = sorted(set(train_adv.columns) & set(test_adv.columns))
    # Chcemy zostawić label, ale nie target / drop_columns
    cols_for_both: List[str] = []
    for c in common_cols:
        if c == AV_LABEL_COL:
            cols_for_both.append(c)
        elif c in drop_columns:
            continue
        else:
            cols_for_both.append(c)

    train_adv = train_adv[cols_for_both].copy()
    test_adv = test_adv[cols_for_both].copy()

    full = pd.concat([train_adv, test_adv], ignore_index=True)

    # Trenujemy klasyfikator "czy to jest test"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label=AV_LABEL_COL,
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(output_dir),
        verbosity=2,
    )

    predictor.fit(
        full,
        presets=presets,
        time_limit=time_limit,
    )

    # Predykcje P(test) tylko dla oryginalnych rekordów train (AV_LABEL_COL == 0)
    mask_train = full[AV_LABEL_COL] == 0
    full_train = full.loc[mask_train].drop(columns=[AV_LABEL_COL])

    proba = predictor.predict_proba(full_train, as_multiclass=False)
    if isinstance(proba, pd.DataFrame):
        # dla binary AutoGluon zwykle zwraca DF z kolumnami {0, 1}
        if 1 in proba.columns:
            av_prob = proba[1]
        else:
            # fallback: bierzemy "najbardziej prawdopodobną" kolumnę
            av_prob = proba.iloc[:, 1]
    else:
        av_prob = proba

    result = pd.DataFrame({
        id_column: train_df[id_column].values,
        AV_PROB_COL: av_prob.values,
    })

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial validation (train vs test).")

    parser.add_argument("--train-path", type=str, required=True, help="Ścieżka do train po preprocesingu.")
    parser.add_argument("--test-path", type=str, required=True, help="Ścieżka do test po preprocesingu.")
    parser.add_argument("--id-column", type=str, required=True, help="Nazwa kolumny id.")
    parser.add_argument("--target-column", type=str, required=False, default=None, help="Nazwa targetu w train.")
    parser.add_argument(
        "--drop-columns",
        type=str,
        nargs="*",
        default=None,
        help="Dodatkowe kolumny do wyrzucenia z cech (np. id).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Gdzie zapisać CSV z av_prob (id_column, av_prob).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Limit czasu na fit adversarial model (sekundy).",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="medium_quality_faster_train",
        help="Preset AutoGluon dla adversarial model.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="autogluon_av_model",
        help="Katalog na model AV (opcionalne, do debugowania).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    output_path = Path(args.output_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    av_df = compute_adversarial_weights(
        train_df=train_df,
        test_df=test_df,
        id_column=args.id_column,
        target_column=args.target_column,
        drop_columns=args.drop_columns,
        presets=args.presets,
        time_limit=args.time_limit,
        output_dir=args.model_output_dir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    av_df.to_csv(output_path, index=False)
    print(f"Saved adversarial weights to: {output_path}")


if __name__ == "__main__":
    main()
