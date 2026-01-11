#!/usr/bin/env python3
"""
Mark stale Optuna RUNNING trials as FAIL in an SQLite storage.

Works across Optuna versions by using:
- storage._storage.set_trial_state_values(...) if available
- fallback to study.tell(..., state=FAIL)

Usage:
  python optuna_clean_zombie_running.py \
    --db /mnt/mlarena/projects/kaggle/playground-series-s6e1/experiments/db/optuna_smoke_s6e1_heavy_v2.sqlite \
    --study smoke_s6e1_heavy_v2 \
    --cutoff-minutes 60 \
    --dry-run

Run without --dry-run to apply changes.
"""

import argparse
from datetime import datetime, timedelta, timezone

import optuna


def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def set_fail(storage: optuna.storages.RDBStorage, study: optuna.Study, trial) -> None:
    """Set trial state to FAIL in a version-tolerant way."""
    trial_id = trial._trial_id

    # Preferred: internal backend API (works for many Optuna versions)
    backend = getattr(storage, "_storage", None)
    if backend is not None:
        fn = getattr(backend, "set_trial_state_values", None)
        if callable(fn):
            # values=None means keep objective values unset; state change only
            fn(trial_id, optuna.trial.TrialState.FAIL, values=None)
            return

    # Fallback: public API
    # study.tell can accept trial number in recent versions; if not, it will error.
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to Optuna SQLite file.")
    p.add_argument("--study", required=True, help="Optuna study name.")
    p.add_argument(
        "--cutoff-minutes",
        type=int,
        default=60,
        help="Mark RUNNING trials older than this as FAIL (default: 60).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print only; no DB changes.")
    p.add_argument("--limit-print", type=int, default=200)
    args = p.parse_args()

    storage_url = f"sqlite:///{args.db}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    study = optuna.load_study(study_name=args.study, storage=storage)

    cutoff = timedelta(minutes=args.cutoff_minutes)
    now = datetime.now(timezone.utc)

    running_count = 0
    candidates = []

    for t in study.get_trials(deepcopy=False):
        if t.state != optuna.trial.TrialState.RUNNING:
            continue
        running_count += 1

        if t.datetime_start is None:
            continue

        start_utc = as_utc(t.datetime_start)
        runtime = now - start_utc

        if runtime > cutoff:
            candidates.append((t, runtime, start_utc))

    print(f"Study: {args.study}")
    print(f"Storage: {storage_url}")
    print(f"Now (UTC): {now.isoformat()}")
    print(f"RUNNING trials: {running_count}")
    print(f"Cutoff: {cutoff}  (>{args.cutoff_minutes} minutes => FAIL)")
    print(f"Candidates to mark FAIL: {len(candidates)}\n")

    candidates.sort(key=lambda x: x[1], reverse=True)
    for t, runtime, start_utc in candidates[: args.limit_print]:
        rt = str(runtime).split(".")[0]
        print(
            f"trial={t.number:6d}  runtime={rt:>10}  "
            f"started_utc={start_utc.isoformat()}  trial_id={t._trial_id}"
        )

    if len(candidates) > args.limit_print:
        print(f"... and {len(candidates) - args.limit_print} more")

    if args.dry_run:
        print("\nDRY RUN: no changes applied.")
        return

    changed = 0
    for t, _runtime, _start_utc in candidates:
        set_fail(storage, study, t)
        changed += 1

    print(f"\nApplied: marked FAIL = {changed}")


if __name__ == "__main__":
    main()

