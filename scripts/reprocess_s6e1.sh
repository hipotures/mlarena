#!/bin/bash
# Parallel batch preprocessing with graceful Ctrl+C handling
PROJECT="playground-series-s6e1"
PREFIX="test_c_01_"
START=1
END=1000
THREADS=16

echo "Starting parallel batch preprocessing for $PROJECT (Threads: $THREADS)..."
echo "Press Ctrl+C to stop starting new tasks and finish current ones."

# Flag to control the loop
keep_running=1

# Trap SIGINT (Ctrl+C) and SIGTERM
trap "echo -e '\n[!] Stop signal received. No new tasks will start. Waiting for current jobs to finish...'; keep_running=0" SIGINT SIGTERM

for i in $(seq -f "%04g" $START $END); do
    # Check if we should stop
    if [ $keep_running -eq 0 ]; then
        break
    fi

    # Manage thread limit: wait if too many background jobs
    while [ $(jobs -rp | wc -l) -ge $THREADS ]; do
        sleep 0.5
        # Re-check stop flag while waiting
        if [ $keep_running -eq 0 ]; then break 2; fi
    done

    # Launch task in background
    (
        TEMPLATE="${PREFIX}${i}"
        uv run python scripts/mla.py preprocess --project "$PROJECT" --preprocess-template "$TEMPLATE" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ $TEMPLATE done"
        else
            # Only print failure if we aren't stopping (to avoid noise)
            if [ $keep_running -eq 1 ]; then
                echo "✗ $TEMPLATE failed"
            fi
        fi
    ) &
done

echo "Waiting for active jobs to complete..."
wait
echo "Batch process finished/stopped."
