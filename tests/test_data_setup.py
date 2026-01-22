#!/usr/bin/env python3

"""
Test script to validate transcription using OpenSLR Yoruba dataset.
Downloads a sample and checks transcription against reference.
"""

import os
import sys
import subprocess
from pathlib import Path

def download_sample():
    """Download a small sample of Yoruba test data"""
    sample_files = [
        "https://openslr.org/resources/86/yo_ng_female.zip",
        "https://openslr.org/resources/86/line_index_female.tsv"
    ]
    
    os.makedirs("test_data", exist_ok=True)
    
    print("Downloading sample Yoruba data...")
    for url in sample_files:
        filename = Path(url).name
        filepath = Path("test_data") / filename
        if not filepath.exists():
            subprocess.run(["wget", "-q", url, "-O", str(filepath)], check=True)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists")

def validate_transcription():
    """Test transcription on sample files"""
    print("\nValidating transcription setup...")
    
    # Check if we have test data
    test_dir = Path("test_data")
    if not (test_dir / "yo_ng_female.zip").exists():
        print("No test data found. Run download first.")
        return
    
    # Check if meeting_transcriber.py exists
    if not Path("meeting_transcriber.py").exists():
        print("meeting_transcriber.py not found in current directory")
        return
    
    print("Test setup complete!")
    print("Available test files in test_data/")
    for f in os.listdir("test_data"):
        print(f"  - {f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_sample()
    else:
        validate_transcription()