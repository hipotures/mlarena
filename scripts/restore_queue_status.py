import json
from pathlib import Path

QUEUE_FILE = Path("projects/kaggle/playground-series-s6e1/queue/queue.json")

if not QUEUE_FILE.exists():
    print("Queue file not found!")
    exit(1)

with open(QUEUE_FILE, "r") as f:
    data = json.load(f)

# IDs that were previously completed (136-154 for templates 100-118)
to_restore = list(range(136, 155))
restored_count = 0

for task in data["queue"]:
    if task["id"] in to_restore:
        task["status"] = "completed"
        restored_count += 1

with open(QUEUE_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(f"Successfully restored {restored_count} tasks to 'completed' status.")
