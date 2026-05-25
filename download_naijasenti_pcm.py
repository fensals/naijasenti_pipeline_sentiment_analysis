
import os
import requests
import time

# Base URL for raw file access
BASE = "https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data"

# All PCM-specific files to download
FILES = {
    # Annotated splits (tweet + label, tab-separated)
    "pcm/train.tsv": "data/raw/pcm_train.tsv",
    "pcm/dev.tsv": "data/raw/pcm_dev.tsv",
    "pcm/test.tsv": "data/raw/pcm_test.tsv",
    # Stopwords
    "stopwords/pcm_stopwords.csv": "data/raw/pcm_stopwords.csv",
    # Lexicons
    "lexicons/manual_sentiment_lexicon_pcm.tsv"  : "data/raw/lexicon_manual.tsv",
    "lexicons/translated_sentiment_pcm.tsv": "data/raw/lexicon_sentiment.tsv",
    "lexicons/translated_emotion_pcm.tsv": "data/raw/lexicon_emotion.tsv",
}


def download_file(url, dest_path):
    """Download a single file from GitHub raw URL with retry logic."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(response.content)
                kb = len(response.content) / 1024
                print(f"  ✓  {dest_path}  ({kb:.1f} KB)")
                return True
            elif response.status_code == 404:
                print(f"  ✗  NOT FOUND: {url}")
                return False
            else:
                print(f"  ⚠  HTTP {response.status_code} — retrying...")
                time.sleep(2)
        except requests.RequestException as e:
            print(f"  ⚠  Connection error ({e}) — retrying...")
            time.sleep(5)
    print(f"  ✗  Failed after 3 attempts: {url}")
    return False


def main():
    print("NaijaSenti PCM Data Downloader")
    print("Source: https://github.com/hausanlp/NaijaSenti")
    print("-" * 60)
    success, fail = 0, 0
    for repo_path, local_path in FILES.items():
        url = f"{BASE}/{repo_path}"
        if download_file(url, local_path):
            success += 1
        else:
            fail += 1
    print("-" * 60)
    print(f"Download complete. Success: {success} | Failed: {fail}")
    if fail > 0:
        print("  Tip: For failed files, check the exact path at:")
        print("  https://github.com/hausanlp/NaijaSenti/tree/main/data")


if __name__ == "__main__":
    main()
