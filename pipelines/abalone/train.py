#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import ast
import pickle
import joblib
import logging
from pathlib import Path

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert(text):
    """Convert JSON string to list of names"""
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def convert_cast(text):
    """Extract top 3 cast members"""
    try:
        L = []
        for i, val in enumerate(ast.literal_eval(text)):
            if i < 3:
                L.append(val['name'])
            else:
                break
        return L
    except:
        return []

def fetch_director(text):
    """Extract director from crew"""
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except:
        return []

def remove_space(L):
    """Remove spaces from list items"""
    return [i.replace(" ", "") for i in L if isinstance(i, str)]

def stems(text):
    """Apply stemming to text"""
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    
    args = parser.parse_args()
    
    logger.info("Starting movie recommendation model training")
    
    # Load the preprocessed data
    train_data_path = os.path.join(args.train, "train.csv")
    logger.info(f"Loading training data from: {train_data_path}")
    
    df = pd.read_csv(train_data_path)
    logger.info(f"Training data shape: {df.shape}")
    
    # Extract features for similarity calculation
    # Assuming the preprocessed data has 'tags' column
    if 'tags' not in df.columns:
        raise ValueError("Training data must contain 'tags' column")
    
    # Vectorization
    logger.info("Creating feature vectors...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    
    # Calculate similarity matrix
    logger.info("Computing similarity matrix...")
    similarity = cosine_similarity(vectors)
    logger.info(f"Similarity matrix shape: {similarity.shape}")
    
    # Save model artifacts
    logger.info(f"Saving model to {args.model_dir}")
    
    with open(os.path.join(args.model_dir, "movie_list.pkl"), "wb") as f:
        pickle.dump(df[['movie_id', 'title']], f)
    
    with open(os.path.join(args.model_dir, "similarity.pkl"), "wb") as f:
        pickle.dump(similarity, f)
    
    # Save vectorizer for potential future use
    joblib.dump(cv, os.path.join(args.model_dir, "vectorizer.pkl"))
    
    logger.info("Model training completed successfully!")
