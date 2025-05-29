
"""Trains the movie recommendation model."""
import argparse
import logging
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting model training.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()
    
    logger.info("Loading training data from %s.", args.train)
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    
    logger.info("Training data shape: %s", train_df.shape)
    logger.info("Creating feature vectors.")
    
    # Create feature vectors
    cv = CountVectorizer(max_features=args.max_features, stop_words='english')
    vectors = cv.fit_transform(train_df['tags']).toarray()
    
    logger.info("Computing similarity matrix.")
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    logger.info("Saving model artifacts to %s.", args.model_dir)
    # Save model artifacts
    with open(os.path.join(args.model_dir, 'movie_list.pkl'), 'wb') as f:
        pickle.dump(train_df, f)
    
    with open(os.path.join(args.model_dir, 'similarity.pkl'), 'wb') as f:
        pickle.dump(similarity_matrix, f)
    
    with open(os.path.join(args.model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(cv, f)
    
    logger.info("Training completed successfully.")
