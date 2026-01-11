#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, timezone

import optuna

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


def get_local_tzinfo():
    # System local tzinfo (whatever the OS/python is configured to)
    return datetime.now().astimezone().tzinfo


def naive_to_utc(dt_naive: datetime, naive_tz: str) -> datetime:
    """Interpret naive datetime in naive_tz and convert to UTC."""
    if naive_tz.lower() == "utc":
        return dt_naive.replace(tzinfo=timezone.utc)

    if naive_tz.lower() == "local":
        tz = get_local_tzinfo()
        return dt_naive.replace(tzinfo=tz).astimezone(timezone.utc)

    # IANA tz name, e.g. Europe/Warsaw
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo not available; use --naive-tz utc or local")
    tz = ZoneInfo(naive_tz)
    return dt_naive.replace(tzinfo=tz).astimezone(timezone.utc)


def as_utc(dt: datetime, now_utc: datetime, naive_tz: str, future_tolerance: timedelta) -> datetime:
    """Convert dt to aware UTC. If dt is naive and naive_tz=auto, choose best interpretation."""
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        return dt.astimezone(timezone.utc)

    # naive
    if naive_tz.lower() != "auto":
        return naive_to_utc(dt, naive_tz)

    # auto: try UTC assumption first
    as_utc_assuming_utc = dt.replace(tzinfo=timezone.utc)

    # if start seems to be in the future, likely naive is local time (e.g. Europe/Warsaw)
    if as_utc_assuming_utc > (now_utc + future_tolerance):
        # prefer Europe/Warsaw if available; fall back to system local
        try:
            if ZoneInfo is not None:
                return naive_to_utc(dt, "Europe/Warsaw")
        except Exception:
            pass
        return naive_to_utc(dt, "local")

    return as_utc_assuming_utc


def set_fail(storage: optuna.storages.RDBStorage, study: optuna.Study, trial) -> None:
    trial_id = trial._trial_id

    backend = getattr(storage, "_storage", None)
    if backend is not None:
        fn = getattr(backend, "set_trial_state_values", None)
        if callable(fn):
            fn(trial_id, optuna.trial.TrialState.FAIL, values=None)
            return

    # fallback
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--study", required=True)
    p.add_argument("--cutoff-minutes", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-print", type=int, default=200)
    p.add_argument(
        "--naive-tz",
        default="auto",
        help='How to interpret naive datetime_start: "auto" (default), "utc", "local", or e.g. "Europe/Warsaw".',
    )
    p.add_argument(
        "--future-tolerance-minutes",
        type=int,
        default=5,
        help="In auto mode, if start time is > now + this tolerance, treat naive as local/Warsaw.",
    )
    args = p.parse_args()

    storage_url = f"sqlite:///{args.db}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    study = optuna.load_study(study_name=args.study, storage=storage)

    cutoff = timedelta(minutes=args.cutoff_minutes)
    now = datetime.now(timezone.utc)
    future_tol = timedelta(minutes=args.future_tolerance_minutes)

    running_count = 0
    candidates = []
    future_starts = 0

    for t in study.get_trials(deepcopy=False):
        if t.state != optuna.trial.TrialState.RUNNING:
            continue
        running_count += 1
        if t.datetime_start is None:
            continue

        start_utc = as_utc(t.datetime_start, now, args.naive_tz, future_tol)
        if start_utc > now + future_tol:
            future_starts += 1

        runtime = now - start_utc
        if runtime > cutoff:
            candidates.append((t, runtime, start_utc))

    print(f"Study: {args.study}")
    print(f"Storage: {storage_url}")
    print(f"Now (UTC): {now.isoformat()}")
    print(f"RUNNING trials: {running_count}")
    print(f"Naive tz mode: {args.naive_tz}  (future tolerance: {future_tol})")
    print(f"Starts considered 'future': {future_starts}")
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
    for t, _, _ in candidates:
        set_fail(storage, study, t)
        changed += 1

    print(f"\nApplied: marked FAIL = {changed}")


if __name__ == "__main__":
    main()

