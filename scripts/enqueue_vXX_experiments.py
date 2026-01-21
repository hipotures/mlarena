import subprocess
import os

PROJECT = "playground-series-s6e1"
START_ID = 100
END_ID = 599

print(f"Adding {START_ID} to {END_ID} experiments to queue for project {PROJECT}...")

for i in range(START_ID, END_ID + 1):
    template = f"test-20250118_{i}"
    
    # Use python scripts/task_queue.py add --model-template ...
    # We want skip_git=true and skip_submit=true
    cmd = [
        "python3", "scripts/task_queue.py",
        "-p", PROJECT,
        "add",
        "--model-template", template,
        "--no-submit",
        "--no-git"
    ]
    
    subprocess.run(cmd, capture_output=True)
    
    if i % 50 == 0:
        print(f"Progress: {i}/599...")

print(f"Done! {END_ID - START_ID + 1} experiments added to the queue.")
