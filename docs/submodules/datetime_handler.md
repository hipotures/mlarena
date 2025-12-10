# Datetime Handler Sub-Module

## Overview

The **datetime_handler** sub-module parses datetime columns, generates derived time features, optionally adds cyclical encodings (sin/cos), and computes time differences between timestamp columns. It is designed to be configurable so you can control how many time features are created and avoid leakage.

**Module Name**: `datetime_handler`  
**Location**: `config/code/preprocessing/datetime_handler.py`

## Capabilities
- Parse specified columns to datetime with optional per-column format.
- Generate basic or extended time features (year, month, day, dayofweek, hour, etc.) or custom sets.
- Cyclical encodings (sin/cos) for periodic features like hour/dayofweek/month/weekofyear.
- Compute time differences between datetime column pairs in chosen units (days/hours/minutes/seconds).
- Optional drop of original datetime columns after expansion.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `datetime_cols` | List[str] | `[]` | Columns to parse as datetime |
| `datetime_formats` | Dict[str,str] | `{}` | Per-column datetime format strings |
| `expand_datetime_cols` | List[str] \| null | `null` | Columns to expand (null = `datetime_cols`) |
| `time_features_set` | str | `"basic"` | `basic`, `extended`, `none`, `custom` |
| `custom_features` | List[str] | `[]` | Features to use when `time_features_set=custom` |
| `cyclical_features` | List[str] | `[]` | Which features to encode cyclically: `hour`, `dayofweek`, `month`, `weekofyear` |
| `time_diff_pairs` | List[dict/list] | `[]` | Pairs to diff. Each entry dict with `start`, `end`, optional `name`, `unit`; or list `[start, end, name?, unit?]` |
| `time_diff_default_unit` | str | `"days"` | Default unit for diffs: `seconds`, `minutes`, `hours`, `days` |
| `drop_original_datetime` | bool | `false` | Drop original datetime cols after expansion |

### Feature Sets
- **basic**: `year, month, day, dayofweek`
- **extended**: basic + `quarter, weekofyear, dayofyear, is_month_start/end, is_quarter_start/end, is_year_start/end, hour, minute`
- **custom**: use `custom_features`
- **none**: no derived features

## Examples

### Basic Expansion
```yaml
datetime_basic:
  module: datetime_handler
  cache: true
  config:
    datetime_cols: ["signup_time"]
    time_features_set: "basic"
```

### Extended + Cyclical
```yaml
datetime_extended_cyc:
  module: datetime_handler
  cache: true
  config:
    datetime_cols: ["event_time"]
    time_features_set: "extended"
    cyclical_features: ["hour", "dayofweek"]
    drop_original_datetime: false
```

### Time Differences
```yaml
datetime_diffs:
  module: datetime_handler
  cache: true
  config:
    datetime_cols: ["signup_time", "last_active"]
    time_diff_pairs:
      - {start: "signup_time", end: "last_active", name: "time_since_signup_days", unit: "days"}
      - ["signup_time", "last_active", "time_since_signup_hours", "hours"]
```

### Custom Features Only
```yaml
datetime_custom:
  module: datetime_handler
  cache: true
  config:
    datetime_cols: ["timestamp"]
    time_features_set: "custom"
    custom_features: ["year", "month", "dayofweek"]
```

## Artifacts
- `datetime_report.json`: parsed columns, derived columns, cyclical columns, time diff columns, config snapshot.
- `summary.json`: standard preprocessing summary (shape/column changes).

## State Dictionary (`fit_transform` return)
```python
{
    "version": "1.0",
    "parsed_columns": ["event_time"],
    "derived_columns": ["event_time_year", "event_time_month", ...],
    "cyclical_columns": ["event_time_hour_sin", "event_time_hour_cos"],
    "time_diff_columns": ["time_since_signup_days"],
    "config": {...}
}
```

## Notes & Tips
- Only columns listed in `datetime_cols` are parsed; expansion defaults to the same list unless `expand_datetime_cols` is set.
- Cyclical encodings require the corresponding derived feature to exist (e.g., `event_time_hour`).
- Time diffs require both columns to be present and parsed as datetime; unit options: seconds/minutes/hours/days.
- Set `drop_original_datetime: true` if you only want derived features and not the raw datetime columns.***
