import argparse
import logging

from src.pipeline import run_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='End-to-End ML Pipeline')
    parser.add_argument('--data_path', type=str, default='data/raw/data.csv',
                       help='Path to the input data file')
    parser.add_argument('--model_path', type=str, default='models/final_model.joblib',
                       help='Path to save the trained model')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with sample data')
    
    args = parser.parse_args()
    
    if args.test:
        # Create sample data for testing
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        sample_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        sample_data['target'] = y
        sample_data.to_csv('data/raw/sample_data.csv', index=False)
        
        logging.info("Created sample data for testing")
        args.data_path = 'data/raw/sample_data.csv'
    
    try:
        logging.info(f"Starting pipeline with data: {args.data_path}")
        model, accuracy = run_pipeline(args.data_path, args.model_path)
        logging.info(f"Pipeline completed successfully. Model accuracy: {accuracy:.4f}")
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()