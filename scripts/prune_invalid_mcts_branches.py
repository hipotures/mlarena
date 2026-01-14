#!/usr/bin/env python3
"""
Prune invalid MCTS branches where child step index <= parent step index.
Requires database with fixed indices (run fix_mcts_db_indices.py first).
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import deque

def prune_database(db_path: Path, dry_run: bool = True):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON") # Ensure cascade delete works
    cur = conn.cursor()
    
    try:
        # 1. Load all edges and nodes to build graph
        print("Loading graph structure...")
        cur.execute("SELECT parent_trial_id, child_trial_id, action_json FROM mcts_edges")
        edges = cur.fetchall()
        
        cur.execute("SELECT trial_id, pipeline_signature FROM mcts_nodes")
        nodes = {row["trial_id"]: row for row in cur.fetchall()}
        
        # Build adjacency list: parent_id -> list of (child_id, step_index)
        adj: Dict[int, List[Any]] = {}
        # Find root(s) - nodes that are never children (except potentially baseline which is depth 0)
        # Actually, baseline usually has parent_trial_id pointing to... nowhere? Or self?
        # In runner logic: parent_trial_id is get_trial_id_by_signature.
        # Baseline is created with depth 0.
        # Let's find root by looking for nodes with depth 0.
        
        cur.execute("SELECT trial_id FROM mcts_nodes WHERE depth=0")
        roots = [row["trial_id"] for row in cur.fetchall()]
        
        if not roots:
            print("Error: No root nodes (depth=0) found!")
            return

        all_child_ids = set()
        
        for edge in edges:
            pid = edge["parent_trial_id"]
            cid = edge["child_trial_id"]
            all_child_ids.add(cid)
            
            try:
                action = json.loads(edge["action_json"])
                idx = action.get("searched_index")
                if idx is None:
                    # Fallback to 'step_index' if searched_index missing
                    idx = action.get("step_index", -1)
            except:
                idx = -1
                
            if pid not in adj: adj[pid] = []
            adj[pid].append({"child_id": cid, "index": idx})

        print(f"Graph loaded: {len(nodes)} nodes, {len(edges)} edges. Roots: {roots}")

        # 2. Traverse and identify invalid trials
        trials_to_delete: Set[int] = set()
        queue = deque() # (trial_id, last_step_index)
        
        # Initialize queue with roots. Roots have last_step_index = -1
        for root_id in roots:
            queue.append((root_id, -1))
            
        visited = set()
        
        while queue:
            curr_id, last_idx = queue.popleft()
            
            if curr_id in visited:
                continue
            visited.add(curr_id)
            
            if curr_id in trials_to_delete:
                # If current node is already marked for deletion (e.g. from a separate bad path leading to it? unlikely in tree), 
                # we don't need to process children, they will be deleted by cascade.
                # But to be safe and explicit, let's process children to mark them too if needed?
                # Actually, if we delete parent, children go away.
                continue

            children = adj.get(curr_id, [])
            for child in children:
                cid = child["child_id"]
                c_idx = child["index"]
                
                # Check validity
                if c_idx <= last_idx:
                    print(f"Invalid edge found: Parent {curr_id} (idx {last_idx}) -> Child {cid} (idx {c_idx}). Pruning branch.")
                    trials_to_delete.add(cid)
                    # We don't add to queue with new index, but we might need to add to queue 
                    # to find *its* children and mark them? 
                    # If we use CASCADE DELETE, we just need to delete the head of the bad branch.
                    # But let's find all descendants just to report count correctly.
                    # BFS for descendants
                    desc_queue = deque([cid])
                    while desc_queue:
                        d_id = desc_queue.popleft()
                        if d_id not in trials_to_delete:
                            trials_to_delete.add(d_id)
                            descendants = adj.get(d_id, [])
                            for desc in descendants:
                                desc_queue.append(desc["child_id"])
                else:
                    queue.append((cid, c_idx))

        print(f"Found {len(trials_to_delete)} invalid trials (and descendants) to prune.")
        
        if not trials_to_delete:
            print("Tree structure is valid. No pruning needed.")
            return

        if dry_run:
            print("Dry run enabled. Use --no-dry-run to execute deletion.")
            return

        # 3. Execute Deletion
        print("Deleting trials...")
        # SQLite limit for IN (...) is usually 999. Batch it.
        to_delete_list = list(trials_to_delete)
        batch_size = 500
        
        total_deleted = 0
        for i in range(0, len(to_delete_list), batch_size):
            batch = to_delete_list[i:i+batch_size]
            placeholders = ",".join("?" for _ in batch)
            # Delete from trials table (Cascade should handle mcts_nodes, mcts_edges, params, values)
            cur.execute(f"DELETE FROM trials WHERE trial_id IN ({placeholders})", batch)
            total_deleted += cur.rowcount
            
        conn.commit()
        print(f"Successfully pruned {total_deleted} trials.")
        
        # Verify orphans (optional)
        cur.execute("SELECT count(*) FROM mcts_nodes WHERE trial_id NOT IN (SELECT trial_id FROM trials)")
        orphans = cur.fetchone()[0]
        if orphans > 0:
            print(f"Warning: {orphans} orphan nodes found (Foreign Keys might be disabled). Cleaning up...")
            cur.execute("DELETE FROM mcts_nodes WHERE trial_id NOT IN (SELECT trial_id FROM trials)")
            conn.commit()

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune invalid MCTS branches.")
    parser.add_argument("db_path", type=Path, help="Path to mcts.db")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually delete records")
    
    args = parser.parse_args()
    
    if not args.db_path.exists():
        print(f"Database file not found: {args.db_path}")
        exit(1)
        
    prune_database(args.db_path, dry_run=not args.no_dry_run)
