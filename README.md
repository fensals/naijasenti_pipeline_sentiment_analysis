## Sentiment Analysis Pipeline

### By: Femi Aleyemi
### Supervised by: Dr Faki Silas.

An end-to-end automated sentiment analysis pipeline for Nigerian Pidgin (PCM) social media text, built as the practical implementation component of my ***PGD Information Technology*** project.
All data is sourced exclusively from the NaijaSenti GitHub repository — no API keys or paid services required.

#### Project Overview
This pipeline classifies Nigerian Pidgin tweets into positive, negative, or neutral sentiment using three models of increasing sophistication:
ModelTypeAccuracyVADERLexicon-based baseline~52%Logistic Regression + TF-IDFTraditional ML (trained on NaijaSenti PCM) about 73% AfriBERTa (fine-tuned)Transformer — African-language pre-trained about 86%


#### Data Source
All data comes from the official NaijaSenti repository:
https://github.com/hausanlp/NaijaSenti/tree/main/data
The download_naijasenti_pcm.py script fetches the following PCM-specific files automatically:

data/pcm/train.tsv — annotated training split (~10,000 tweets)
data/pcm/dev.tsv — validation split (~2,000 tweets)
data/pcm/test.tsv — held-out test split (~2,000 tweets)
data/stopwords/pcm_stopwords.csv — native-speaker-curated Pidgin stopwords
data/lexicons/manual_sentiment_lexicon_pcm.tsv — manually annotated Pidgin sentiment lexicon
data/lexicons/translated_sentiment_pcm.tsv — translated Afinn sentiment lexicon
data/lexicons/translated_emotion_pcm.tsv — translated emotion lexicon


#### Getting Started
1. Clone the repository
bashgit clone https://github.com/fensals/naijasenti_pipeline_sentiment_analysis.git
cd naijasenti_pipeline_sentiment_analysis
2. Set up the environment
bashbash setup_environment.sh
conda activate naijasenti_pcm
3. Run the pipeline in order

#### Step 1 — Download data from NaijaSenti GitHub
python download_naijasenti_pcm.py

#### Step 2 — Explore the dataset
python explore_pcm_dataset.py

#### Step 3 — Preprocess with PySpark
python preprocess_pcm.py

#### Step 4 — Run classifiers
python classify_vader.py
python classify_tfidf_lr.py
python classify_afriberta.py   # GPU recommended

#### Step 5 — Evaluate and compare
python evaluate_all_models.py

#### Step 6 — Hypothesis test
python hypothesis_test.py

#### Requirements

Python 3.11
Apache Spark 3.5 / PySpark
PyTorch 2.2+
HuggingFace Transformers 4.39+
scikit-learn, pandas, matplotlib, seaborn, scipy

Install everything via setup_environment.sh or:
bashpip install -r requirements.txt

#### Citation
If you use this work, please also cite the NaijaSenti dataset:

Muhammad, S. H., Adelani, D. I., Ruder, S., Ahmad, I. S., Abdulmumin, I., Bello, B. S., Choudhury, M., Emezue, C. C., Abdullahi, S. S., Aremu, A., Jorge, A., & Brazdil, P. (2022). NaijaSenti: A Nigerian Twitter sentiment corpus for multilingual sentiment analysis. Proceedings of the Thirteenth Language Resources and Evaluation Conference, 590-602.


