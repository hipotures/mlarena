# Submission Queue Guide

**Purpose:** Batch management of Kaggle submission uploads with duplicate detection and error tracking.

**Important:** This is a **separate system** from the Task Queue (`mla queue`). See [Comparison Table](#submission-queue-vs-task-queue) below.

---

## Overview

The Submission Queue is a standalone script (`scripts/submission_queue.py`) that manages the upload of prediction files to Kaggle. It provides:

- **Duplicate detection** - Prevents re-uploading identical submissions
- **Batch processing** - Queue multiple submissions and upload later
- **Error tracking** - Logs all upload attempts with timestamps
- **Auto-cleanup** - Removes successfully uploaded submissions
- **Thread-safe** - Multiple processes can safely access the queue

---

## Basic Usage

### 1. Add Submission to Queue

During the submit step, use `submit.queue_submit=true` to add to queue instead of immediate upload:

```bash
uv run python scripts/mla.py submit \
  project=<competition-slug> \
  experiment_id=<experiment_id> \
  submit.queue_submit=true
```

**What happens:**
- Submission file copied to `submissions/` directory
- Entry added to `submissions/queue.json`
- Status set to `pending`
- No upload occurs yet

---

### 2. List Queued Submissions

View all pending and failed submissions:

```bash
python scripts/submission_queue.py --project <competition-slug> list
```

**Example output:**
```
Submission Queue for project: titanic

Queue #  | Status    | Experiment ID           | Filename                      | Added At
---------|-----------|-------------------------|-------------------------------|--------------------
1        | pending   | exp-20251226-103504     | submission-20251226103520.csv | 2025-12-26 10:35:20
2        | pending   | exp-20251226-110345     | submission-20251226110402.csv | 2025-12-26 11:04:02
3        | failed    | exp-20251225-152430     | submission-20251225152445.csv | 2025-12-25 15:24:45
```

---

### 3. Submit from Queue

Upload a queued submission to Kaggle:

**By queue number:**
```bash
python scripts/submission_queue.py --project <competition-slug> submit 1
```

**By experiment ID:**
```bash
python scripts/submission_queue.py --project <competition-slug> submit exp-20251226-103504
```

**By filename:**
```bash
python scripts/submission_queue.py --project <competition-slug> submit submission-20251226103520.csv
```

**What happens:**
1. Checks Kaggle API for duplicate (same filename already uploaded)
2. If duplicate: skips upload, marks as completed
3. If new: uploads to Kaggle via `kaggle competitions submit`
4. Updates status to `submitted`
5. Logs attempt in queue file

---

### 4. Submit with Auto Fetch-Score

Upload and automatically fetch the public score after 30 seconds:

```bash
python scripts/submission_queue.py --project <competition-slug> submit 1 --continue-flow
```

**What happens:**
1. Uploads submission (as above)
2. Waits 30 seconds for Kaggle processing
3. Runs `mla fetch-score` to scrape public score
4. If successful: removes entry from queue
5. If failed: keeps in queue as `failed` for retry

**Use case:** Overnight batch processing - queue multiple submissions during the day, run with `--continue-flow` at night to upload and fetch all scores.

---

### 5. Remove from Queue

Remove a submission without uploading:

```bash
python scripts/submission_queue.py --project <competition-slug> remove 1
```

**Use case:** Remove submissions that are no longer needed (e.g., inferior results, experiment cancelled).

---

## Queue File Structure

Queue data stored in: `projects/kaggle/<competition-slug>/submissions/queue.json`

**Format:**
```json
{
  "queue": [
    {
      "queue_number": 1,
      "experiment_id": "exp-20251226-103504",
      "filename": "submission-20251226103520.csv",
      "status": "pending",
      "added_at": "2025-12-26 10:35:20",
      "submitted_at": null,
      "error": null,
      "attempts": []
    }
  ]
}
```

**Status values:**
- `pending` - Queued, not yet uploaded
- `submitted` - Successfully uploaded to Kaggle
- `completed` - Uploaded and score fetched (only with `--continue-flow`)
- `failed` - Upload attempt failed (check `error` field)

---

## Features

### Duplicate Detection

Before uploading, the script checks Kaggle API for existing submissions with the same filename:

```python
# Internal logic (simplified):
existing = kaggle.api.competitions_submissions_list(competition)
if submission.csv in [s.fileName for s in existing]:
    print("Duplicate detected, skipping upload")
    mark_as_completed()
```

**Benefit:** Saves API quota and avoids duplicate entries on Kaggle leaderboard.

---

### Error Tracking

All upload attempts are logged with timestamps and error messages:

```json
{
  "attempts": [
    {
      "timestamp": "2025-12-26 10:45:30",
      "success": false,
      "error": "403 Forbidden - Daily submission limit reached"
    }
  ]
}
```

**Use case:** Debug upload failures, track rate limits, audit submission history.

---

### Thread Safety

The queue uses file locking to prevent concurrent modifications:

```python
# Multiple processes can safely call:
submission_queue.py submit 1  # Process A
submission_queue.py submit 2  # Process B (waits for lock)
```

**Benefit:** Safe to run multiple uploads in parallel or use queue from scripts.

---

## Common Workflows

### Workflow 1: Batch Upload at End of Day

```bash
# During experiments:
uv run python scripts/mla.py submit project=titanic experiment_id=exp1 submit.queue_submit=true
uv run python scripts/mla.py submit project=titanic experiment_id=exp2 submit.queue_submit=true
uv run python scripts/mla.py submit project=titanic experiment_id=exp3 submit.queue_submit=true

# End of day: upload all at once
python scripts/submission_queue.py --project titanic submit 1
python scripts/submission_queue.py --project titanic submit 2
python scripts/submission_queue.py --project titanic submit 3
```

---

### Workflow 2: Overnight Auto-Upload with Scores

```bash
# Queue submissions during the day
uv run python scripts/mla.py submit project=titanic experiment_id=exp1 submit.queue_submit=true
# ... more experiments ...

# Create overnight script: upload_all.sh
#!/bin/bash
for i in {1..10}; do
  python scripts/submission_queue.py --project titanic submit $i --continue-flow
  sleep 60  # Wait between uploads to respect rate limits
done
```

**Run overnight:**
```bash
nohup bash upload_all.sh > upload_log.txt 2>&1 &
```

---

### Workflow 3: Selective Upload After Review

```bash
# Queue multiple experiments
# ... run experiments with submit.queue_submit=true ...

# Review queue
python scripts/submission_queue.py --project titanic list

# Upload only the best ones
python scripts/submission_queue.py --project titanic submit exp-20251226-110345
python scripts/submission_queue.py --project titanic submit exp-20251226-143022

# Remove the rest
python scripts/submission_queue.py --project titanic remove 1
python scripts/submission_queue.py --project titanic remove 3
```

---

## Submission Queue vs Task Queue

MLArena has **two separate queue systems** for different purposes:

| Feature | **Submission Queue** | **Task Queue** |
|:--------|:---------------------|:---------------|
| **Purpose** | Upload predictions to Kaggle | Run experiments (train models, preprocess) |
| **Command** | `python scripts/submission_queue.py` | `mla queue` (or `uv run python scripts/mla.py queue`) |
| **Script** | `scripts/submission_queue.py` | `scripts/task_queue.py` |
| **Queue File** | `submissions/queue.json` | `queue/queue.json` |
| **Scope** | `submit` and `fetch-score` modules only | Full pipeline (`preprocess`, `model`, `predict`, etc.) |
| **When to Use** | Batch upload multiple submissions | Queue multiple training jobs |
| **Parallelization** | Manual (user runs script multiple times) | Sequential execution via `mla queue run` |
| **Typical Use Case** | Upload 10 submissions at night | Train 20 model variants overnight |

### When to Use Which

**Use Submission Queue when:**
- You have multiple predictions ready to upload
- You want to avoid daily submission limits by spreading uploads
- You need duplicate detection before upload
- You want to review predictions before uploading

**Use Task Queue when:**
- You want to train multiple models sequentially
- You need to run different preprocessing configurations
- You want to queue experiments for overnight execution
- You need priority-based execution

**Use both together:**
```bash
# Phase 1: Queue training jobs
uv run python scripts/mla.py queue project=titanic add model_template=cpu-best-8h --priority 1
uv run python scripts/mla.py queue project=titanic add model_template=gpu-dev-5m --priority 2

# Phase 2: Run task queue (trains models, generates predictions, queues submissions)
uv run python scripts/mla.py queue project=titanic run

# Phase 3: Upload queued submissions with scores
for i in {1..5}; do
  python scripts/submission_queue.py --project titanic submit $i --continue-flow
done
```

---

## Troubleshooting

### "Queue file not found"

**Cause:** No submissions have been queued yet.

**Fix:** Queue at least one submission:
```bash
uv run python scripts/mla.py submit project=titanic experiment_id=eda submit.queue_submit=true
```

---

### "Submission file not found"

**Cause:** Submission CSV was moved or deleted.

**Fix:** Re-run predict and submit:
```bash
uv run python scripts/mla.py predict project=titanic experiment_id=<experiment_id>
uv run python scripts/mla.py submit project=titanic experiment_id=<experiment_id> submit.queue_submit=true
```

---

### "Duplicate detected, skipping"

**Cause:** Submission with same filename already uploaded to Kaggle.

**Fix:** This is normal behavior (duplicate detection working). If you want to force upload:
1. Rename the file manually
2. Update queue entry
3. Submit again

Or: Simply remove from queue if upload not needed.

---

### "Daily submission limit reached"

**Cause:** Kaggle limits submissions per day (usually 5-10).

**Fix:** Wait until next day, submissions remain in queue with `failed` status. Retry:
```bash
python scripts/submission_queue.py --project titanic submit 1
```

---

## Advanced Usage

### Script Integration

Use submission queue in your own scripts:

```python
import json
from pathlib import Path

# Read queue
queue_file = Path("projects/kaggle/titanic/submissions/queue.json")
with open(queue_file) as f:
    queue_data = json.load(f)

# Find pending submissions
pending = [q for q in queue_data["queue"] if q["status"] == "pending"]
print(f"Pending uploads: {len(pending)}")

# Submit via CLI
import subprocess
for item in pending:
    subprocess.run([
        "python", "scripts/submission_queue.py",
        "--project", "titanic",
        "submit", str(item["queue_number"])
    ])
```

---

### Monitoring Queue Status

Check queue status programmatically:

```bash
# Count pending submissions
python scripts/submission_queue.py --project titanic list | grep pending | wc -l

# Get latest queue entry
cat projects/kaggle/titanic/submissions/queue.json | jq '.queue[-1]'
```

---

## See Also

- **Task Queue**: [README.md - Task Queue Management](../README.md#task-queue-management)
- **Submit Module**: [README.md - Automated Submission](../README.md#automated-submission--score-fetching)
- **Fetch Score**: [README.md - Control Flags](../README.md#control-flags)
- **Auto-Flow**: [README.md - Auto-Flow](../README.md#auto-flow-recommended)
