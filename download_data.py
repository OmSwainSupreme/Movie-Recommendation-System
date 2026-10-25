#!/usr/bin/env python
"""
Dataset download script for the Movie Recommendation System.

Downloads and extracts the MovieLens ml-latest-small dataset from GroupLens.
Run this script once to set up the data directory.

Usage:
    python download_data.py
"""

import os
import sys
import zipfile
import shutil
import ssl
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
EXTRACT_DIR = DATA_DIR / "ml-latest-small"
TARGET_FILES = ["movies.csv", "ratings.csv"]


def print_status(message: str) -> None:
    """Print a status message with formatting."""
    print(f"[INFO] {message}")


def print_error(message: str) -> None:
    """Print an error message with formatting."""
    print(f"[ERROR] {message}", file=sys.stderr)


def print_success(message: str) -> None:
    """Print a success message with formatting."""
    print(f"[SUCCESS] {message}")


def check_existing_data() -> bool:
    """Check if the required data files already exist."""
    for filename in TARGET_FILES:
        if not (DATA_DIR / filename).exists():
            return False
    return True


def download_dataset() -> bool:
    """Download the MovieLens dataset zip file."""
    print_status(f"Downloading dataset from {DATASET_URL}...")
    
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        # Create SSL context that doesn't verify certificates (for Windows environments with cert issues)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Use urlopen with custom SSL context, then write to file
        print_status("Downloading with custom SSL context...")
        with urlopen(DATASET_URL, context=ssl_context) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 8192
            with open(ZIP_PATH, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = min(100, (downloaded * 100) // total_size)
                        sys.stdout.write(f"\r  Progress: {percent}%")
                        sys.stdout.flush()
        print()  # New line after progress
        
        if ZIP_PATH.stat().st_size == 0:
            print_error("Downloaded file is empty")
            return False
            
        print_success(f"Downloaded {ZIP_PATH.stat().st_size:,} bytes")
        return True
        
    except HTTPError as e:
        print_error(f"HTTP error downloading dataset: {e.code} {e.reason}")
        return False
    except URLError as e:
        print_error(f"Network error downloading dataset: {e.reason}")
        return False
    except Exception as e:
        print_error(f"Unexpected error downloading dataset: {e}")
        return False


def extract_dataset() -> bool:
    """Extract the required CSV files from the zip archive."""
    print_status("Extracting dataset...")
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # List contents for debugging
            zip_contents = zip_ref.namelist()
            print_status(f"Zip contents: {zip_contents}")
            
            # Extract only the files we need
            for filename in TARGET_FILES:
                # Find the file in the zip (it's inside ml-latest-small/ folder)
                zip_path = f"ml-latest-small/{filename}"
                if zip_path not in zip_contents:
                    print_error(f"{filename} not found in zip archive")
                    return False
                
                # Extract to data directory
                zip_ref.extract(zip_path, DATA_DIR)
                
                # Move from extracted subdirectory to data directory
                extracted_path = DATA_DIR / zip_path
                target_path = DATA_DIR / filename
                
                if extracted_path.exists():
                    shutil.move(str(extracted_path), str(target_path))
                    print_status(f"Extracted {filename}")
                else:
                    print_error(f"Failed to extract {filename}")
                    return False
            
            # Clean up the extracted directory
            if EXTRACT_DIR.exists():
                shutil.rmtree(EXTRACT_DIR)
                
        return True
        
    except zipfile.BadZipFile:
        print_error("Downloaded file is not a valid zip archive")
        return False
    except Exception as e:
        print_error(f"Error extracting dataset: {e}")
        return False


def verify_dataset() -> bool:
    """Verify that the required files exist and have expected columns."""
    print_status("Verifying dataset...")
    
    for filename in TARGET_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print_error(f"Missing file: {filename}")
            return False
        
        # Quick validation - check file is not empty and has headers
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                if not header:
                    print_error(f"File is empty: {filename}")
                    return False
                print_status(f"  {filename}: {header}")
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    header = f.readline().strip()
                    print_status(f"  {filename} (latin-1): {header}")
            except Exception as e:
                print_error(f"Cannot read {filename}: {e}")
                return False
        except Exception as e:
            print_error(f"Error reading {filename}: {e}")
            return False
    
    return True


def cleanup_zip() -> None:
    """Remove the downloaded zip file to save space."""
    if ZIP_PATH.exists():
        try:
            ZIP_PATH.unlink()
            print_status("Cleaned up zip file")
        except Exception:
            pass  # Ignore cleanup errors


def main() -> int:
    """Main download and setup routine."""
    print("=" * 60)
    print("MovieLens ml-latest-small Dataset Download")
    print("=" * 60)
    
    # Check if data already exists
    if check_existing_data():
        print_success("Dataset files already exist in data/ directory")
        print_status("Skipping download. Delete data/*.csv to re-download.")
        return 0
    
    # Download
    if not download_dataset():
        return 1
    
    # Extract
    if not extract_dataset():
        return 1
    
    # Verify
    if not verify_dataset():
        return 1
    
    # Cleanup
    cleanup_zip()
    
    print("=" * 60)
    print_success("Dataset setup complete!")
    print_status(f"Files ready in {DATA_DIR}/:")
    for filename in TARGET_FILES:
        filepath = DATA_DIR / filename
        size = filepath.stat().st_size
        print_status(f"  {filename} ({size:,} bytes)")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())