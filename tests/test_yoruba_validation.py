#!/usr/bin/env python3

"""
Basic test using OpenSLR Yoruba data to validate transcription workflow.
"""

import os
import subprocess
from pathlib import Path

def test_sample_transcription():
    """Run transcription on sample Yoruba audio file"""
    print("Testing transcription workflow with OpenSLR Yoruba data...")
    
    # Test with a sample file
    audio_file = "test_data/yof_00295_00020329077.wav"
    transcript_file = "test_data/line_index_female.tsv"
    
    if not Path(audio_file).exists():
        print(f"Sample audio not found: {audio_file}")
        return False
    
    # Get reference transcript
    reference = None
    if Path(transcript_file).exists():
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0] == Path(audio_file).name:
                    reference = parts[1]
                    break
    
    print(f"Reference transcript: {reference}")
    
    # Run transcription (simulate - would call actual transcriber)
    print(f"Would transcribe: {audio_file}")
    print("Expected steps:")
    print("  1. Load audio file")
    print("  2. Run speech-to-text")
    print("  3. Compare with reference")
    
    return True

def test_speaker_identification():
    """Test speaker identification (male vs female)"""
    print("\nTesting speaker identification...")
    
    female_files = list(Path("test_data").glob("yof_*.wav"))
    male_files = list(Path("test_data").glob("yom_*.wav"))
    
    print(f"Found {len(female_files)} female files")
    print(f"Found {len(male_files)} male files")
    
    # Sample male file for comparison
    male_sample = "test_data/yom_00295_00012717311.wav"
    if Path(male_sample).exists():
        print(f"Male sample available: {male_sample}")
        print("Speaker labels:")
        print("  - yof_* = Female speaker")
        print("  - yom_* = Male speaker")
    
    return len(female_files) > 0 and len(male_files) > 0

def test_annotations():
    """Test if annotation handling works"""
    print("\nTesting annotation handling...")
    
    annotations = {
        "[breath]": "Audible breathing or sniff",
        "[external]": "External audio interference", 
        "[hesitation]": "Speaker hesitation",
        "[snap]": "Mouth/tongue movement sounds",
        "[abrupt]": "Sudden audio cutoff"
    }
    
    print("Supported annotations:")
    for ann, desc in annotations.items():
        print(f"  {ann}: {desc}")
    
    # Check if any transcripts have annotations
    if Path("test_data/line_index_female.tsv").exists():
        with open("test_data/line_index_female.tsv", 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if '[' in line and ']' in line:
                    count += 1
            print(f"Found {count} annotated utterances")

def main():
    """Run all tests"""
    print("=== OpenSLR Yoruba Test Suite ===")
    print(f"Python: {subprocess.run(['python', '--version'], capture_output=True, text=True).stdout.strip()}")
    
    # Change to test directory
    os.chdir("test_data")
    
    success = True
    success &= test_sample_transcription()
    success &= test_speaker_identification()
    test_annotations()
    
    if success:
        print("\n✓ Test environment ready!")
        print("Next: Integrate with meeting_transcriber.py transcription pipeline")
    else:
        print("\n✗ Test setup incomplete")

if __name__ == "__main__":
    main()