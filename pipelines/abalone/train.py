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
import subprocess
import sys
# Install required libraries via pip
required_packages = ['pandas', 'numpy', 'nltk', 'scikit-learn']
for package in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Download necessary NLTK data
import nltk
nltk.download('punkt')


from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#!/usr/bin/env python3

import pickle
import json
import numpy as np
import logging
import os
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir: str) -> Tuple[Any, Any]:
    """
    Load the model artifacts from the model directory
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Tuple containing (movie_list, similarity_matrix)
    """
    logger.info(f"Loading model from directory: {model_dir}")
    
    try:
        # Load movie list
        with open(os.path.join(model_dir, "movie_list.pkl"), "rb") as f:
            movie_list = pickle.load(f)
        
        # Load similarity matrix
        with open(os.path.join(model_dir, "similarity.pkl"), "rb") as f:
            similarity = pickle.load(f)
        
        logger.info(f"Loaded movie list with {len(movie_list)} movies")
        logger.info(f"Loaded similarity matrix of shape: {similarity.shape}")
        
        return movie_list, similarity
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def input_fn(request_body: str, request_content_type: str = "application/json") -> Dict[str, Any]:
    """
    Parse input data for inference
    
    Args:
        request_body: The request body as a string
        request_content_type: The content type of the request
        
    Returns:
        Parsed input data as dictionary
    """
    logger.info(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        try:
            input_data = json.loads(request_body)
            logger.info(f"Parsed input data: {input_data}")
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data: Dict[str, Any], model: Tuple[Any, Any]) -> List[str]:
    """
    Generate movie recommendations
    
    Args:
        input_data: Dictionary containing movie name and optional parameters
        model: Tuple containing (movie_list, similarity_matrix)
        
    Returns:
        List of recommended movie titles
    """
    movie_list, similarity = model
    
    try:
        # Extract movie name from input
        movie_name = input_data.get('movie')
        if not movie_name:
            raise ValueError("Missing 'movie' field in input data")
        
        # Get number of recommendations (default: 5)
        num_recommendations = input_data.get('num_recommendations', 5)
        
        logger.info(f"Getting recommendations for movie: {movie_name}")
        logger.info(f"Number of recommendations requested: {num_recommendations}")
        
        # Find the movie index
        movie_matches = movie_list[movie_list['title'].str.lower() == movie_name.lower()]
        
        if movie_matches.empty:
            # Try partial matching if exact match fails
            movie_matches = movie_list[movie_list['title'].str.contains(movie_name, case=False, na=False)]
            
        if movie_matches.empty:
            logger.error(f"Movie '{movie_name}' not found in database")
            return {"error": f"Movie '{movie_name}' not found in database"}
        
        # Get the index of the first matching movie
        movie_index = movie_matches.index[0]
        logger.info(f"Found movie at index: {movie_index}")
        
        # Get similarity scores for this movie
        distances = list(enumerate(similarity[movie_index]))
        
        # Sort by similarity (descending) and exclude the movie itself
        distances = sorted(distances, reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        
        # Get recommended movie titles
        recommendations = []
        for i, score in distances:
            movie_title = movie_list.iloc[i]['title']
            recommendations.append({
                'title': movie_title,
                'similarity_score': float(score)
            })
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

def output_fn(prediction: List[str], content_type: str = "application/json") -> str:
    """
    Format the prediction output
    
    Args:
        prediction: List of recommended movies or error dict
        content_type: Output content type
        
    Returns:
        JSON formatted response
    """
    logger.info(f"Formatting output with content type: {content_type}")
    
    if content_type == "application/json":
        try:
            response = {
                "recommendations": prediction if isinstance(prediction, list) else [],
                "status": "success" if isinstance(prediction, list) else "error",
                "message": prediction.get("error", "") if isinstance(prediction, dict) else "Recommendations generated successfully"
            }
            return json.dumps(response, indent=2)
        except Exception as e:
            logger.error(f"Error formatting output: {str(e)}")
            return json.dumps({"error": f"Output formatting failed: {str(e)}"})
    else:
        raise ValueError(f"Unsupported output content type: {content_type}")
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
