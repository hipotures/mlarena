#!/usr/bin/env python3
"""
RENAME FILES WITH CREATION/MODIFICATION DATES
==============================================

This script renames all Python files and markdown files in the current directory
to include their creation/modification date as a prefix.

Format: YYYYMMDD_HHMM_original_filename.ext
Example: 20250104_2013_xgboost_analysis.py

Author: Claude
Date: 2025-01-04
"""

import os
import re
from datetime import datetime
from pathlib import Path
import shutil

def get_file_date(filepath):
    """
    Get file creation or modification date.
    Uses modification time as it's more reliable across platforms.
    """
    stat = os.stat(filepath)
    # Use modification time (more reliable than creation time)
    timestamp = stat.st_mtime
    return datetime.fromtimestamp(timestamp)

def create_new_filename(original_path, file_date):
    """
    Create new filename with date prefix.
    """
    path = Path(original_path)
    original_name = path.name
    
    # Check if file already has date prefix (YYYYMMDD_HHMM_)
    date_pattern = r'^\d{8}_\d{4}_'
    if re.match(date_pattern, original_name):
        print(f"  Skipping {original_name} - already has date prefix")
        return None
    
    # Format date as YYYYMMDD_HHMM
    date_prefix = file_date.strftime('%Y%m%d_%H%M')
    
    # Create new filename
    new_name = f"{date_prefix}_{original_name}"
    new_path = path.parent / new_name
    
    return new_path

def rename_files(directory=".", extensions=['.py', '.md'], dry_run=True):
    """
    Rename files with date prefixes.
    
    Args:
        directory: Directory to process (default: current directory)
        extensions: List of file extensions to process
        dry_run: If True, only show what would be renamed without actually doing it
    """
    directory = Path(directory)
    
    print(f"{'DRY RUN - ' if dry_run else ''}Processing files in: {directory.absolute()}")
    print(f"Looking for extensions: {extensions}")
    print("-" * 80)
    
    # Collect all files to rename
    files_to_rename = []
    
    for ext in extensions:
        for filepath in directory.glob(f"*{ext}"):
            if filepath.is_file():
                file_date = get_file_date(filepath)
                new_path = create_new_filename(filepath, file_date)
                
                if new_path:
                    files_to_rename.append({
                        'original': filepath,
                        'new': new_path,
                        'date': file_date
                    })
    
    # Sort by date (oldest first)
    files_to_rename.sort(key=lambda x: x['date'])
    
    # Display what will be renamed
    if files_to_rename:
        print(f"\nFound {len(files_to_rename)} files to rename:\n")
        
        for item in files_to_rename:
            date_str = item['date'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {item['original'].name}")
            print(f"  → {item['new'].name}")
            print(f"    (modified: {date_str})")
            print()
        
        if not dry_run:
            print("\nRenaming files...")
            renamed_count = 0
            
            for item in files_to_rename:
                try:
                    # Check if target already exists
                    if item['new'].exists():
                        print(f"  WARNING: {item['new'].name} already exists - skipping")
                        continue
                    
                    # Rename file
                    item['original'].rename(item['new'])
                    print(f"  ✓ Renamed: {item['original'].name} → {item['new'].name}")
                    renamed_count += 1
                    
                except Exception as e:
                    print(f"  ✗ Error renaming {item['original'].name}: {e}")
            
            print(f"\nSuccessfully renamed {renamed_count} files.")
        else:
            print("\n[DRY RUN MODE] No files were actually renamed.")
            print("Run with dry_run=False to perform actual renaming.")
    else:
        print("\nNo files found to rename.")

def create_backup(directory=".", backup_name="backup_before_rename"):
    """
    Create a backup of all Python and Markdown files before renaming.
    """
    directory = Path(directory)
    backup_dir = directory / backup_name
    
    print(f"Creating backup in: {backup_dir}")
    
    # Create backup directory
    backup_dir.mkdir(exist_ok=True)
    
    # Copy files
    file_count = 0
    for ext in ['.py', '.md']:
        for filepath in directory.glob(f"*{ext}"):
            if filepath.is_file() and filepath.parent == directory:
                dest = backup_dir / filepath.name
                shutil.copy2(filepath, dest)
                file_count += 1
    
    print(f"Backed up {file_count} files to {backup_dir}")
    return backup_dir

def main():
    """
    Main function with safety checks.
    """
    print("FILE RENAMING SCRIPT - Add Date Prefixes")
    print("=" * 80)
    
    # Safety warning
    print("\n⚠️  WARNING: This script will rename files in the current directory!")
    print("It's recommended to:")
    print("1. Run with dry_run=True first (default)")
    print("2. Create a backup before running")
    print("3. Run on a copy of your files")
    
    # Configuration
    directory = "."  # Current directory
    extensions = ['.py', '.md']  # File types to rename
    
    # First, show what would be renamed (dry run)
    print("\n" + "=" * 80)
    print("PREVIEW MODE (Dry Run)")
    print("=" * 80)
    rename_files(directory, extensions, dry_run=True)
    
    # Ask user if they want to proceed
    print("\n" + "=" * 80)
    response = input("\nDo you want to:\n"
                    "  1. Create backup and rename files\n"
                    "  2. Rename files without backup\n"
                    "  3. Exit without changes\n"
                    "Choose (1/2/3): ").strip()
    
    if response == "1":
        # Create backup first
        backup_dir = create_backup(directory)
        print(f"\n✓ Backup created in: {backup_dir}")
        
        # Perform actual renaming
        print("\n" + "=" * 80)
        print("PERFORMING ACTUAL RENAME")
        print("=" * 80)
        rename_files(directory, extensions, dry_run=False)
        
    elif response == "2":
        # Rename without backup
        confirm = input("\n⚠️  Are you sure you want to rename without backup? (yes/no): ").strip().lower()
        if confirm == "yes":
            print("\n" + "=" * 80)
            print("PERFORMING ACTUAL RENAME")
            print("=" * 80)
            rename_files(directory, extensions, dry_run=False)
        else:
            print("\nOperation cancelled.")
    
    else:
        print("\nOperation cancelled. No files were renamed.")

if __name__ == "__main__":
    # For testing/development, you can also call specific functions:
    # rename_files(".", ['.py', '.md'], dry_run=True)  # Preview only
    # rename_files(".", ['.py', '.md'], dry_run=False)  # Actually rename
    
    main()  # Interactive mode with safety checks