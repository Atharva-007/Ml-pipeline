import unittest

import numpy as np
import pandas as pd

from src.data_processing import preprocess_data
from src.model import evaluate_model, train_model


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_preprocessing(self):
        X_train, X_test, y_train, y_test, _ = preprocess_data(self.sample_data)
        self.assertEqual(len(X_train), 4)  # 80% of 5 samples
    
    def test_model_training(self):
        X_train = np.array([[1, 0.1], [2, 0.2], [3, 0.3]])
        y_train = np.array([0, 1, 0])
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()