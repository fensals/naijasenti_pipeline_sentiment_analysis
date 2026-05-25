
import re
import unicodedata
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType

# SparkSession Initialisation 
spark = SparkSession.builder \
    .appName("NaijaSenti_PCM_Preprocessing") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

SILVER_DIR = "data/silver"

#Load NaijaSenti PCM Stopwords 
# The stopword list is from: data/stopwords/pcm_stopwords.csv
# It was curated by native Pidgin speakers as part of the NaijaSenti project.
sw_df = pd.read_csv("data/raw/pcm_stopwords.csv")
PCM_STOPWORDS = set(sw_df.iloc[:, 0].str.strip().str.lower().tolist())
# Augment with standard English stopwords that are not sentiment-bearing
EXTRA_SW = {"the","a","an","is","are","was","were","in","on","at","to","of",
            "and","or","but","for","with","it","this","that","as","be","by",
            "from","not","so","if","then","when","where","who","what","how",
            "@user","rt"}
ALL_STOPWORDS = PCM_STOPWORDS | EXTRA_SW
print(f"Total stopwords loaded: {len(ALL_STOPWORDS)}")

#  Load PCM Sentiment Lexicon 
# The manual sentiment lexicon provides a curated vocabulary of positive and
# negative Pidgin words, used here as a reference during EDA and feature
# engineering (not during cleaning to avoid removing sentiment-bearing tokens).
try:
    lex_df = pd.read_csv("data/raw/lexicon_manual.tsv", sep="\t",
                         names=["word","sentiment"], header=None)
    PCM_POS_WORDS = set(lex_df[lex_df["sentiment"]=="positive"]["word"].tolist())
    PCM_NEG_WORDS = set(lex_df[lex_df["sentiment"]=="negative"]["word"].tolist())
    print(f"Lexicon loaded: {len(PCM_POS_WORDS)} positive | {len(PCM_NEG_WORDS)} negative words")
except Exception as e:
    print(f"Warning: lexicon not loaded ({e}). Proceeding without it.")
    PCM_POS_WORDS, PCM_NEG_WORDS = set(), set()

# Nigerian Pidgin Normalisation Dictionary 
# Maps Pidgin orthographic variants to normalised tokens, preserving semantic
# intent while improving model generalisation.
PIDGIN_NORM = {
    r"\bdem\b"      : "them",
    r"\bdis\b"      : "this",
    r"\bdat\b"      : "that",
    r"\bna\b"       : "is",
    r"\bno be\b"    : "is not",
    r"\bwetin\b"    : "what",
    r"\babi\b"      : "or is it",
    r"\bsabi\b"     : "know",
    r"\bwahala\b"   : "trouble",
    r"\bchop\b"     : "eat",
    r"\bdey\b"      : "is are",
    r"\buna\b"      : "you all",
    r"\bshey\b"     : "is it right",
    r"\bginger\b"   : "motivate",
    r"\btank\b"     : "thank",
    r"\bno cap\b"   : "seriously",
    r"\bsapa\b"     : "poverty broke",
    r"\bjapa\b"     : "emigrate leave",
    r"\bsebi\b"     : "is it not",
    r"\bkpele\b"    : "sorry sympathy",
    r"\bbe like say\b": "it seems that",
    r"\bgo fit\b"   : "can will be able",
    r"\bno go\b"    : "should not will not",
    r"\bwella\b"    : "very much indeed",
    r"\bnow now\b"  : "immediately right now",
}

# Emoji → Sentiment Word Map 
EMOJI_MAP = {
    "\U0001F602": "very funny laughing",
    "\U0001F621": "very angry furious",
    "\U0001F62D": "crying sad",
    "\U0001F44D": "good positive",
    "\U0001F44E": "bad negative disapprove",
    "\U0001F525": "excellent great fire",
    "\U0001F614": "tired disappointed",
    "\U0001F64F": "prayer hope",
    "\u2764"     : "love care",
    "\U0001F4B8" : "money wealth",
}


def clean_pidgin_tweet(text):
    """
    Full preprocessing pipeline for a single Nigerian Pidgin tweet.
    Steps:
      1. Guard against null / non-string inputs
      2. Lowercase
      3. Map emojis to sentiment words
      4. Remove pre-anonymised @USER placeholders (redundant noise)
      5. Strip residual RT artefacts
      6. Apply Pidgin normalisation dictionary
      7. Preserve hashtag text (remove # symbol, keep word)
      8. Remove non-ASCII artefacts
      9. Remove punctuation except apostrophe and hyphen
      10. Collapse whitespace and trim
      11. Filter trivially short results
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = text.lower()                                          # Step 2
    for emoji, words in EMOJI_MAP.items():                      # Step 3
        text = text.replace(emoji, f" {words} ")
    text = re.sub(r"@user", " ", text)                         # Step 4
    text = re.sub(r"^rt\s*:?\s*", "", text)                  # Step 5
    for pattern, replacement in PIDGIN_NORM.items():            # Step 6
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"#(\w+)", r" \1 ", text)                   # Step 7
    text = unicodedata.normalize("NFKD", text)                  # Step 8
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)              # Step 9
    text = re.sub(r"\s+", " ", text).strip()                   # Step 10
    return text if len(text.split()) >= 2 else ""                # Step 11


def remove_stopwords(text):
    """Remove stopwords using the NaijaSenti PCM stopword list."""
    if not text:
        return ""
    tokens = [w for w in text.split() if w not in ALL_STOPWORDS]
    return " ".join(tokens) if tokens else ""


# Register PySpark UDFs
clean_udf = F.udf(clean_pidgin_tweet, StringType())
sw_udf    = F.udf(remove_stopwords,   StringType())


def process_split(tsv_path, split_name):
    """Load, clean, and save one annotated split as Parquet."""
    print(f"\nProcessing {split_name} split: {tsv_path}")

    # Load TSV — two columns: tweet, label (no header in NaijaSenti files)
    df = spark.read.csv(
        tsv_path, sep="\t", header=False,
        schema="tweet STRING, label STRING"
    )
    raw_count = df.count()
    print(f"  Loaded  : {raw_count} rows")

    # Drop nulls
    df = df.dropna(subset=["tweet", "label"])
    # Keep only valid labels
    df = df.filter(F.col("label").isin(["positive", "negative", "neutral"]))
    # Remove duplicates on tweet text
    df = df.dropDuplicates(["tweet"])
    print(f"  After QC: {df.count()} rows")

    # Apply cleaning
    df = df.withColumn("cleaned_text", clean_udf(F.col("tweet")))
    df = df.filter(F.col("cleaned_text") != "")
    # Remove stopwords
    df = df.withColumn("processed_text", sw_udf(F.col("cleaned_text")))
    df = df.withColumn("split", F.lit(split_name))

    final_count = df.count()
    print(f"  Final   : {final_count} rows after cleaning")

    # Save to Silver layer as Parquet
    out_path = f"{SILVER_DIR}/{split_name}"
    df.select("tweet", "cleaned_text", "processed_text", "label", "split") \
      .write.mode("overwrite").parquet(out_path)
    print(f"  Saved   : {out_path}")
    df.show(5, truncate=80)
    return final_count


if __name__ == "__main__":
    import os
    os.makedirs(SILVER_DIR, exist_ok=True)
    n_train = process_split("data/raw/pcm_train.tsv", "train")
    n_dev   = process_split("data/raw/pcm_dev.tsv",   "dev")
    n_test  = process_split("data/raw/pcm_test.tsv",  "test")
    print(f"\nPreprocessing complete. Train:{n_train} | Dev:{n_dev} | Test:{n_test}")
