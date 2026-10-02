"""Download and extract MultiWOZ 2.4 dataset."""

import json
import os
import urllib.request
import zipfile

MULTIWOZ_URL = "https://github.com/smartyfh/MultiWOZ2.4/archive/refs/heads/main.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_multiwoz():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "multiwoz24.zip")

    if os.path.exists(zip_path):
        print("Already downloaded.")
    else:
        print("Downloading MultiWOZ 2.4...")
        urllib.request.urlretrieve(MULTIWOZ_URL, zip_path)
        print("Done.")

    extract_dir = os.path.join(RAW_DIR, "MultiWOZ2.4-main")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        print("Extracted to", extract_dir)
    else:
        print("Already extracted.")

    return extract_dir


if __name__ == "__main__":
    download_multiwoz()
