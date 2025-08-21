import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {
            'RandomForest': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC
        }
    
    def train_model(self, X_train, y_train):
        """Train the selected model"""
        model_name = self.config['model']['algorithm']
        model_params = self.config['model']['parameters']
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported")
        
        model = self.models[model_name](**model_params)
        model.fit(X_train, y_train)
        
        logging.info(f"Trained {model_name} model with parameters: {model_params}")
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1': f1_score(y_test, predictions, average='weighted')
        }
        
        if self.config['evaluation']['classification_report']:
            report = classification_report(y_test, predictions, output_dict=True)
            metrics['classification_report'] = report
        
        if self.config['evaluation']['confusion_matrix']:
            cm = confusion_matrix(y_test, predictions)
            self.plot_confusion_matrix(cm, model.__class__.__name__)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(self.config['output']['plots_path'], exist_ok=True)
        plt.savefig(f"{self.config['output']['plots_path']}confusion_matrix_{model_name}.png")
        plt.close()
        
        logging.info(f"Confusion matrix saved to {self.config['output']['plots_path']}confusion_matrix_{model_name}.png")
    
    def cross_validate(self, model, X, y):
        """Perform cross-validation"""
        cv_scores = cross_val_score(
            model, X, y, 
            cv=self.config['training']['cv_folds'],
            scoring=self.config['training']['scoring']
        )
        
        return {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'all_scores': cv_scores.tolist()
        }
    
    def save_model(self, model, filename):
        """Save trained model"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(model, filename)
        logging.info(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load trained model"""
        return joblib.load(filename)