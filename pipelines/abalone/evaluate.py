"""Evaluates the movie recommendation model."""
import argparse
import json
import logging
import os
import pickle
import pandas as pd
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def get_recommendations(movie_name, movie_list, similarity_matrix, top_k=5):
    """Get movie recommendations."""
    try:
        movie_matches = movie_list[movie_list['title'].str.contains(movie_name, case=False, na=False)]
        if movie_matches.empty:
            return []
        
        index = movie_matches.index[0]
        distances = list(enumerate(similarity_matrix[index]))
        distances = sorted(distances, reverse=True, key=lambda x: x[1])[1:top_k+1]
        
        return [movie_list.iloc[i[0]].title for i in distances]
    except:
        return []

def calculate_diversity(recommendations_list):
    """Calculate diversity of recommendations."""
    all_recommendations = [rec for recs in recommendations_list for rec in recs]
    unique_recommendations = len(set(all_recommendations))
    total_recommendations = len(all_recommendations)
    
    if total_recommendations == 0:
        return 0.0
    
    return unique_recommendations / total_recommendations

if __name__ == "__main__":
    logger.info("Starting model evaluation.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    logger.info("Loading model artifacts from %s.", args.model_dir)
    # Load model artifacts
    with open(os.path.join(args.model_dir, 'movie_list.pkl'), 'rb') as f:
        movie_list = pickle.load(f)
    
    with open(os.path.join(args.model_dir, 'similarity.pkl'), 'rb') as f:
        similarity_matrix = pickle.load(f)
    
    logger.info("Loading test data from %s.", args.test_data)
    test_df = pd.read_csv(os.path.join(args.test_data, "test.csv"))
    
    logger.info("Running evaluation on %d test samples.", len(test_df))
    
    # Sample movies for evaluation
    sample_movies = test_df['title'].sample(min(50, len(test_df))).tolist()
    
    # Get recommendations for sample movies
    all_recommendations = []
    successful_recommendations = 0
    
    for movie in sample_movies:
        recommendations = get_recommendations(movie, movie_list, similarity_matrix)
        if recommendations:
            all_recommendations.append(recommendations)
            successful_recommendations += 1
    
    # Calculate metrics
    diversity_score = calculate_diversity(all_recommendations)
    coverage_score = len(movie_list) / len(movie_list)  # 100% coverage for content-based
    recommendation_success_rate = successful_recommendations / len(sample_movies)
    
    evaluation_results = {
        "diversity_score": diversity_score,
        "coverage_score": coverage_score,
        "recommendation_success_rate": recommendation_success_rate,
        "total_movies_in_catalog": len(movie_list),
        "test_sample_size": len(sample_movies),
        "successful_recommendations": successful_recommendations
    }
    
    logger.info("Evaluation Results: %s", evaluation_results)
    
    # Save evaluation results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info("Evaluation completed successfully.")
