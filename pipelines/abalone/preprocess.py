"""Feature engineers the movie dataset for recommendation system."""
import argparse
import logging
import os
import pathlib
import boto3
import numpy as np
import pandas as pd
import ast
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def convert_features(text):
    """Convert string representation of list to actual list of names."""
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def convert_cast(text, limit=3):
    """Extract top cast members."""
    try:
        L = []
        for i, val in enumerate(ast.literal_eval(text)):
            if i < limit:
                L.append(val['name'])
            else:
                break
        return L
    except:
        return []

def fetch_director(text):
    """Extract director from crew."""
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
    except:
        pass
    return []

def remove_space(L):
    """Remove spaces from list items."""
    return [i.replace(" ", "") for i in L]

def stem_text(text, stemmer):
    """Apply Porter stemming to text."""
    return " ".join([stemmer.stem(i) for i in text.split()])

if __name__ == "__main__":
    logger.info("Starting movie data preprocessing.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test").mkdir(parents=True, exist_ok=True)
    
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key_prefix = "/".join(input_data.split("/")[3:])
    
    logger.info("Downloading data from bucket: %s, key prefix: %s", bucket, key_prefix)
    
    # Download movies and credits data
    s3 = boto3.resource("s3")
    movies_fn = f"{base_dir}/data/tmdb_5000_movies.csv"
    credits_fn = f"{base_dir}/data/tmdb_5000_credits.csv"
    
    s3.Bucket(bucket).download_file(f"{key_prefix}/tmdb_5000_movies.csv", movies_fn)
    s3.Bucket(bucket).download_file(f"{key_prefix}/tmdb_5000_credits.csv", credits_fn)
    
    logger.info("Reading downloaded data.")
    movies = pd.read_csv(movies_fn)
    credits = pd.read_csv(credits_fn)
    
    # Clean up downloaded files
    os.unlink(movies_fn)
    os.unlink(credits_fn)
    
    logger.info("Merging datasets and selecting features.")
    # Merge datasets
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    
    logger.info("Applying feature transformations.")
    # Process features
    movies['genres'] = movies['genres'].apply(convert_features)
    movies['keywords'] = movies['keywords'].apply(convert_features)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # Remove spaces from categorical features
    movies['cast'] = movies['cast'].apply(remove_space)
    movies['crew'] = movies['crew'].apply(remove_space)
    movies['genres'] = movies['genres'].apply(remove_space)
    movies['keywords'] = movies['keywords'].apply(remove_space)
    
    # Create tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create final dataset
    processed_df = movies[['movie_id', 'title', 'tags']].copy()
    processed_df['tags'] = processed_df['tags'].apply(lambda x: " ".join(x).lower())
    
    # Apply stemming
    logger.info("Applying stemming to text features.")
    ps = PorterStemmer()
    processed_df['tags'] = processed_df['tags'].apply(lambda x: stem_text(x, ps))
    
    logger.info("Splitting data into train and test sets.")
    # Split data (80% train, 20% test)
    train_data = processed_df.sample(frac=0.8, random_state=42)
    test_data = processed_df.drop(train_data.index)
    
    logger.info("Writing processed datasets to %s.", base_dir)
    # Save processed data
    train_data.to_csv(f"{base_dir}/train/train.csv", index=False)
    test_data.to_csv(f"{base_dir}/test/test.csv", index=False)
    
    logger.info("Preprocessing completed successfully.")
