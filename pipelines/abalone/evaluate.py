#!/usr/bin/env python3

import json
import os
import pickle
import pandas as pd
import numpy as np
import logging
import tarfile
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_model_artifacts(model_path):
    """
    Extract model artifacts from tar.gz file if present
    Returns the path to extracted artifacts
    """
    tar_file_path = os.path.join(model_path, "model.tar.gz")
    
    if os.path.exists(tar_file_path):
        logger.info(f"Found model.tar.gz file at {tar_file_path}")
        
        # Create extraction directory
        extract_dir = os.path.join(model_path, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract tar.gz file
        try:
            with tarfile.open(tar_file_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
                logger.info(f"Successfully extracted model artifacts to {extract_dir}")
                
                # List extracted files for debugging
                extracted_files = os.listdir(extract_dir)
                logger.info(f"Extracted files: {extracted_files}")
                
            return extract_dir
            
        except Exception as e:
            logger.error(f"Failed to extract model.tar.gz: {str(e)}")
            raise
    
    else:
        # If no tar.gz file, assume artifacts are already extracted
        logger.info("No model.tar.gz found, assuming artifacts are already extracted")
        return model_path

def load_model_artifacts(artifacts_path):
    """
    Load model artifacts from the given path
    """
    similarity_path = os.path.join(artifacts_path, "similarity.pkl")
    movie_list_path = os.path.join(artifacts_path, "movie_list.pkl")
    
    # Check if required files exist
    if not os.path.exists(similarity_path):
        raise FileNotFoundError(f"similarity.pkl not found at {similarity_path}")
    
    if not os.path.exists(movie_list_path):
        raise FileNotFoundError(f"movie_list.pkl not found at {movie_list_path}")
    
    # Load similarity matrix
    logger.info(f"Loading similarity matrix from {similarity_path}")
    with open(similarity_path, "rb") as f:
        similarity_matrix = pickle.load(f)
    
    # Load movie list
    logger.info(f"Loading movie list from {movie_list_path}")
    with open(movie_list_path, "rb") as f:
        movie_list = pickle.load(f)
    
    return similarity_matrix, movie_list

def evaluate_recommendations(similarity_matrix, test_df, top_k=5):
    """
    Evaluate the recommendation system using various metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['total_movies'] = len(test_df)
    metrics['similarity_matrix_shape'] = similarity_matrix.shape
    metrics['average_similarity'] = float(np.mean(similarity_matrix))
    metrics['similarity_std'] = float(np.std(similarity_matrix))
    
    # Coverage metrics
    non_zero_similarities = np.count_nonzero(similarity_matrix)
    total_possible_pairs = similarity_matrix.shape[0] * similarity_matrix.shape[1]
    metrics['coverage'] = float(non_zero_similarities / total_possible_pairs)
    
    # Diversity metrics (average pairwise distance of top recommendations)
    diversity_scores = []
    for i in range(min(100, len(test_df))):  # Sample 100 movies for efficiency
        # Get top k similar movies
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[-top_k-1:-1]  # Exclude self
        
        if len(top_indices) > 1:
            # Calculate average pairwise distance in top recommendations
            top_similarities = similarities[top_indices]
            avg_diversity = 1 - np.mean(top_similarities)
            diversity_scores.append(avg_diversity)
    
    if diversity_scores:
        metrics['average_diversity'] = float(np.mean(diversity_scores))
        metrics['diversity_std'] = float(np.std(diversity_scores))
    else:
        metrics['average_diversity'] = 0.0
        metrics['diversity_std'] = 0.0
    
    # Recommendation quality score (composite metric)
    # Higher similarity variance suggests better discrimination
    similarity_variance = np.var(similarity_matrix, axis=1)
    metrics['average_discrimination'] = float(np.mean(similarity_variance))
    
    # Calculate a composite quality score
    # Good recommendation system should have:
    # - High coverage
    # - Good diversity
    # - Good discrimination ability
    quality_score = (
        metrics['coverage'] * 0.3 + 
        metrics['average_diversity'] * 0.4 + 
        metrics['average_discrimination'] * 0.3
    )
    metrics['quality_score'] = float(quality_score)
    
    return metrics

if __name__ == "__main__":
    logger.info("Starting model evaluation")
    
    try:
        # Model path
        model_path = "/opt/ml/processing/model"
        
        # Extract model artifacts if needed
        artifacts_path = extract_model_artifacts(model_path)
        
        # Load model artifacts
        similarity_matrix, movie_list = load_model_artifacts(artifacts_path)
        
        logger.info(f"Loaded similarity matrix of shape: {similarity_matrix.shape}")
        logger.info(f"Loaded movie list with {len(movie_list)} movies")
        
        # Load test data
        test_data_path = "/opt/ml/processing/test/test.csv"
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}")
        
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data with {len(test_df)} movies")
        
        # Validate that similarity matrix dimensions match the data
        if similarity_matrix.shape[0] != len(movie_list):
            logger.warning(f"Similarity matrix rows ({similarity_matrix.shape[0]}) != movie list length ({len(movie_list)})")
        
        # Evaluate the model
        evaluation_metrics = evaluate_recommendations(similarity_matrix, test_df)
        
        # Log key metrics
        logger.info("Evaluation Results:")
        logger.info(f"Quality Score: {evaluation_metrics['quality_score']:.4f}")
        logger.info(f"Coverage: {evaluation_metrics['coverage']:.4f}")
        logger.info(f"Average Diversity: {evaluation_metrics['average_diversity']:.4f}")
        logger.info(f"Average Discrimination: {evaluation_metrics['average_discrimination']:.4f}")
        
        # Prepare evaluation report
        report_dict = {
            "recommendation_metrics": evaluation_metrics,
            "model_quality": {
                "quality_score": evaluation_metrics['quality_score'],
                "status": "PASS" if evaluation_metrics['quality_score'] > 0.3 else "FAIL"
            },
            "model_info": {
                "similarity_matrix_shape": list(similarity_matrix.shape),
                "movie_count": len(movie_list),
                "test_data_count": len(test_df)
            }
        }
        
        # Save evaluation report
        output_dir = "/opt/ml/processing/evaluation"
        os.makedirs(output_dir, exist_ok=True)
        
        evaluation_path = os.path.join(output_dir, "evaluation.json")
        with open(evaluation_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Evaluation report saved to {evaluation_path}")
        logger.info("Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        
        # Create failure report
        failure_report = {
            "recommendation_metrics": {},
            "model_quality": {
                "quality_score": 0.0,
                "status": "FAIL",
                "error": str(e)
            }
        }
        
        output_dir = "/opt/ml/processing/evaluation"
        os.makedirs(output_dir, exist_ok=True)
        
        evaluation_path = os.path.join(output_dir, "evaluation.json")
        with open(evaluation_path, "w") as f:
            json.dump(failure_report, f, indent=2)
        
        raise