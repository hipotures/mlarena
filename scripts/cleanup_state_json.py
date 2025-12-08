#!/usr/bin/env python3
"""
Clean up bloated state.json files by removing heavy ydata-profiling keys.

Usage:
    python cleanup_state_json.py /path/to/experiments/
    python cleanup_state_json.py /path/to/experiments/ --dry-run
"""
import json
import sys
from pathlib import Path
from typing import Any

# Keys to remove from ydata profiling output
PROFILE_REMOVE_KEYS = {
    "value_counts_without_nan",
    "value_counts_index_sorted",
    "histogram",
    "histogram_length",
    "character_counts",
    "block_alias_values",
    "category_alias_values",
    "block_alias_char_counts",
    "script_char_counts",
    "category_alias_char_counts",
    "package",
    "analysis",
    "time_index_analysis",
}
WORD_COUNT_LIMIT = 50


def sanitize_payload(payload: Any) -> Any:
    """Recursively remove heavy keys from nested dict/list structure."""
    def _clean(node: Any) -> Any:
        if isinstance(node, dict):
            cleaned = {}
            for key, value in node.items():
                if key in PROFILE_REMOVE_KEYS:
                    continue
                if key == "word_counts" and isinstance(value, dict):
                    cleaned[key] = value if len(value) <= WORD_COUNT_LIMIT else {}
                    continue
                cleaned[key] = _clean(value)
            return cleaned
        if isinstance(node, list):
            return [_clean(item) for item in node]
        return node

    return _clean(payload)


def cleanup_state_file(state_path: Path, dry_run: bool = False) -> dict:
    """
    Clean a single state.json file.

    Returns dict with stats: {"removed_keys": int, "size_before": int, "size_after": int}
    """
    try:
        size_before = state_path.stat().st_size

        with open(state_path) as f:
            data = json.load(f)

        # Count keys before cleaning (rough estimate)
        data_str = json.dumps(data)
        keys_before = sum(1 for key in PROFILE_REMOVE_KEYS if f'"{key}"' in data_str)

        # Clean
        cleaned = sanitize_payload(data)

        # Write back if not dry run
        if not dry_run:
            with open(state_path, 'w') as f:
                json.dump(cleaned, f, indent=2)

            size_after = state_path.stat().st_size
        else:
            # Estimate size after
            size_after = len(json.dumps(cleaned))

        return {
            "removed_keys": keys_before,
            "size_before": size_before,
            "size_after": size_after,
            "saved_bytes": size_before - size_after,
            "saved_pct": ((size_before - size_after) / size_before * 100) if size_before > 0 else 0,
        }

    except Exception as e:
        print(f"  ✗ Error processing {state_path}: {e}", file=sys.stderr)
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up bloated state.json files by removing heavy ydata-profiling data"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to search for state.json files (recursive)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files"
    )
    parser.add_argument(
        "--pattern",
        default="state.json",
        help="Filename pattern to match (default: state.json)"
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"✗ Directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Find all state.json files
    state_files = list(args.directory.rglob(args.pattern))

    if not state_files:
        print(f"✗ No {args.pattern} files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(state_files)} {args.pattern} file(s)")
    if args.dry_run:
        print("🔍 DRY RUN MODE - no files will be modified\n")
    else:
        print()

    total_saved = 0
    total_before = 0
    total_after = 0
    processed = 0

    for state_path in sorted(state_files):
        rel_path = state_path.relative_to(args.directory)
        print(f"Processing: {rel_path}")

        stats = cleanup_state_file(state_path, dry_run=args.dry_run)

        if stats:
            processed += 1
            total_before += stats["size_before"]
            total_after += stats["size_after"]
            total_saved += stats["saved_bytes"]

            print(f"  Size: {stats['size_before']:,} → {stats['size_after']:,} bytes")
            print(f"  Saved: {stats['saved_bytes']:,} bytes ({stats['saved_pct']:.1f}%)")

            if stats["removed_keys"] > 0:
                print(f"  Removed {stats['removed_keys']} heavy key(s)")

            if not args.dry_run:
                print("  ✓ Cleaned")
            print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {processed}/{len(state_files)} files")
    print(f"Total size before: {total_before / 1024 / 1024:.1f} MB")
    print(f"Total size after:  {total_after / 1024 / 1024:.1f} MB")
    print(f"Total saved:       {total_saved / 1024 / 1024:.1f} MB ({(total_saved/total_before*100) if total_before else 0:.1f}%)")

    if args.dry_run:
        print("\n⚠ DRY RUN - Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
