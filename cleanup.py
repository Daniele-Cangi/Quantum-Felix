#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Felix Cleanup Script
============================

This script helps you delete generated files and build artifacts.
Usage: python cleanup.py [options]

This answers the question "come lo elimino?" (how do I delete it?)
"""

import os
import shutil
import argparse
import glob
from pathlib import Path


def remove_path(path, description="", dry_run=False):
    """Remove a file or directory with error handling"""
    if not os.path.exists(path):
        print(f"  ✓ {description}: already clean")
        return
        
    try:
        if dry_run:
            print(f"  → Would delete {description}: {path}")
            return
            
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  ✓ Deleted {description}: {path}")
        else:
            os.remove(path)
            print(f"  ✓ Deleted {description}: {path}")
    except Exception as e:
        print(f"  ✗ Failed to delete {description}: {e}")


def cleanup_python_cache(dry_run=False):
    """Remove Python cache files and directories"""
    print("🐍 Cleaning Python cache files...")
    
    # Remove __pycache__ directories
    for pycache_dir in glob.glob("**/__pycache__", recursive=True):
        remove_path(pycache_dir, "Python cache directory", dry_run)
    
    # Remove .pyc files
    for pyc_file in glob.glob("**/*.pyc", recursive=True):
        remove_path(pyc_file, "Python bytecode file", dry_run)
    
    # Remove .pyo files  
    for pyo_file in glob.glob("**/*.pyo", recursive=True):
        remove_path(pyo_file, "Python optimized bytecode file", dry_run)


def cleanup_build_artifacts(dry_run=False):
    """Remove build and distribution artifacts"""
    print("🔨 Cleaning build artifacts...")
    
    # Simple directories
    simple_artifacts = [
        ("build/", "Build directory"),
        ("dist/", "Distribution directory"), 
        (".eggs/", "Eggs directory"),
    ]
    
    for pattern, description in simple_artifacts:
        remove_path(pattern, description, dry_run)
    
    # Pattern-based artifacts
    for egg_info_dir in glob.glob("**/*.egg-info", recursive=True):
        remove_path(egg_info_dir, "Egg info directory", dry_run)
    
    for egg_file in glob.glob("**/*.egg", recursive=True):
        remove_path(egg_file, "Egg file", dry_run)


def cleanup_quantum_felix_files(dry_run=False):
    """Remove Quantum Felix specific generated files"""
    print("⚡ Cleaning Quantum Felix generated files...")
    
    # Search for these files in common locations
    quantum_files = [
        "effective_config.json",
        "complex_steps.csv", 
        "complex_summary.json",
    ]
    
    processed_files = set()
    
    for file_pattern in quantum_files:
        # Check exact path first
        if os.path.exists(file_pattern) and file_pattern not in processed_files:
            remove_path(file_pattern, "Quantum Felix generated file", dry_run)
            processed_files.add(file_pattern)
        
        # Search recursively
        for found_file in glob.glob(f"**/{os.path.basename(file_pattern)}", recursive=True):
            if found_file not in processed_files:
                remove_path(found_file, "Quantum Felix generated file", dry_run)
                processed_files.add(found_file)


def cleanup_logs_and_temp(dry_run=False):
    """Remove log files and temporary files"""
    print("📝 Cleaning logs and temporary files...")
    
    # Log files
    for log_file in glob.glob("**/*.log", recursive=True):
        remove_path(log_file, "Log file", dry_run)
    
    # Temporary files
    temp_patterns = ["**/*~", "**/*.tmp", "**/*.temp", "**/.DS_Store"]
    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern, recursive=True):
            remove_path(temp_file, "Temporary file", dry_run)


def cleanup_ide_files(dry_run=False):
    """Remove IDE and editor files"""
    print("💻 Cleaning IDE files...")
    
    ide_dirs = [".vscode/", ".idea/", ".spyderproject/", ".ropeproject/"]
    for ide_dir in ide_dirs:
        remove_path(ide_dir, "IDE directory", dry_run)
    
    # Editor temp files
    for pattern in ["**/*.swp", "**/*.swo"]:
        for temp_file in glob.glob(pattern, recursive=True):
            remove_path(temp_file, "Editor temporary file", dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up Quantum Felix generated files and build artifacts",
        epilog="This script answers 'come lo elimino?' (how do I delete it?)"
    )
    parser.add_argument(
        "--dry-run", "-n", 
        action="store_true",
        help="Show what would be deleted without actually deleting anything"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true", 
        help="Clean everything (default)"
    )
    parser.add_argument(
        "--python-only", "-p",
        action="store_true",
        help="Only clean Python cache files"
    )
    parser.add_argument(
        "--build-only", "-b", 
        action="store_true",
        help="Only clean build artifacts"
    )
    parser.add_argument(
        "--quantum-only", "-q",
        action="store_true", 
        help="Only clean Quantum Felix specific files"
    )
    
    args = parser.parse_args()
    
    print("🧹 Quantum Felix Cleanup Tool")
    print("=" * 40)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print()
    
    # Determine what to clean
    clean_all = args.all or not any([args.python_only, args.build_only, args.quantum_only])
    
    if clean_all or args.python_only:
        cleanup_python_cache(args.dry_run)
        print()
    
    if clean_all or args.build_only:
        cleanup_build_artifacts(args.dry_run)
        print()
    
    if clean_all or args.quantum_only:
        cleanup_quantum_felix_files(args.dry_run)
        print()
    
    if clean_all:
        cleanup_logs_and_temp(args.dry_run)
        print()
        cleanup_ide_files(args.dry_run)
        print()
    
    if args.dry_run:
        print("🔍 Dry run complete. Use without --dry-run to actually delete files.")
    else:
        print("✅ Cleanup complete!")
    
    print("\n📖 How to delete (come eliminare):")
    print("  • Run: python cleanup.py (delete everything)")
    print("  • Run: python cleanup.py --dry-run (preview what will be deleted)")
    print("  • Run: python cleanup.py --python-only (delete only Python cache)")
    print("  • Run: python cleanup.py --build-only (delete only build artifacts)")
    print("  • Run: python cleanup.py --quantum-only (delete only Quantum Felix files)")


if __name__ == "__main__":
    main()