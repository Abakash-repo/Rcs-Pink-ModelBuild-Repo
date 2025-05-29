"""Feature engineering script for movie recommendation preprocessing."""
import argparse
import logging
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Install and setup NLTK
try:
    import nltk
except ImportError:
    print("Installing NLTK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')  
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def preprocess_text(text):
    """Preprocess text data for movie recommendation."""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and stem
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    processed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            processed_tokens.append(ps.stem(token))
    
    return ' '.join(processed_tokens)

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    base_dir = "/opt/ml/processing"
    logger.info("Reading input data from %s.", args.input_data)
    
    # Read the input data
    df = pd.read_csv(args.input_data)
    logger.info("Input data shape: %s", df.shape)
    
    # Preprocess the data - adjust column names based on your dataset
    # Assuming your movie dataset has columns like 'title', 'overview', 'genres', etc.
    
    # Example preprocessing for movie recommendation:
    # Combine relevant text features into 'tags' column
    if 'overview' in df.columns:
        df['overview'] = df['overview'].fillna('')
    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('')
    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].fillna('')
    
    # Create tags column by combining text features
    df['tags'] = df.get('overview', '') + ' ' + df.get('genres', '') + ' ' + df.get('keywords', '')
    
    # Apply text preprocessing
    logger.info("Applying text preprocessing...")
    df['tags'] = df['tags'].apply(preprocess_text)
    
    # Remove rows with empty tags
    df = df[df['tags'].str.strip() != '']
    
    logger.info("Processed data shape: %s", df.shape)
    
    # Split the data
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    logger.info("Train size: %s, Validation size: %s, Test size: %s", 
                len(train_df), len(val_df), len(test_df))
    
    # Save the splits
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "validation")
    test_dir = os.path.join(base_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(train_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(val_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(test_dir, "test.csv"), index=False)
    
    logger.info("Preprocessing completed successfully.")
