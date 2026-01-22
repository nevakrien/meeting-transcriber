"""
Test fixtures and utilities for meeting-transcriber tests.
"""
import os
import tempfile
from pathlib import Path

# Get the absolute path to the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"

def get_fixture_path(relative_path: str) -> Path:
    """Get absolute path to a fixture file."""
    return FIXTURES_DIR / relative_path

def temp_audio_file(suffix: str = ".wav") -> str:
    """Create a temporary audio file path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path