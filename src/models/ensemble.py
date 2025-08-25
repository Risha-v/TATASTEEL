import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingRegressor, 
    BaggingRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from config.paths import Paths
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EnsemblePredictor:
    """
    Enhanced ensemble predictor with multiple ensemble techniques
    """
    
    def __init__(self):
        self.ensemble_model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_metrics = {}
        self.is_trained = False
        self.logger = get_logger(__name__)
    
    def train(self, X, y, preprocessor=None):
        """Train ensemble model with comprehensive evaluation"""
        self.logger.info("Starting Ensemble Predictor training...")
        
        try:
            self.preprocessor = preprocessor
            
            # Create base models
            base_models = self._create_base_models()
            
            # Create ensemble
            self.ensemble_model = VotingRegressor(
                estimators=[(name, model) for name, model in base_models.items()],
                n_jobs=-1
            )
            
            # Train ensemble
            if preprocessor:
                # Create pipeline with preprocessor
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('ensemble', self.ensemble_model)
                ])
                pipeline.fit(X, y)
                self.ensemble_model = pipeline
            else:
                self.ensemble_model.fit(X, y)
            
            # Evaluate performance
            self._evaluate_performance(X, y)
            
            self.is_trained = True
            self.logger.info("Ensemble training completed successfully")
            
            return self.ensemble_model
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {str(e)}")
            raise
    
    def _create_base_models(self):
        """Create base models for ensemble"""
        models = {
            'rf': RandomForestRegressor(**Config.RANDOM_FOREST_PARAMS),
            'xgb': XGBRegressor(**Config.XGBOOST_PARAMS),
            'lgb': LGBMRegressor(**Config.LIGHTGBM_PARAMS),
            'cat': CatBoostRegressor(**Config.CATBOOST_PARAMS),
            'gbr': GradientBoostingRegressor(**Config.GRADIENT_BOOSTING_PARAMS)
        }
        return models
    
    def _evaluate_performance(self, X, y):
        """Evaluate ensemble performance using cross-validation"""
        try:
            from sklearn.model_selection import cross_val_score
            
            # Cross-validation metrics
            r2_scores = cross_val_score(
                self.ensemble_model, X, y, 
                cv=Config.CV_FOLDS, 
                scoring='r2', 
                n_jobs=-1
            )
            
            rmse_scores = cross_val_score(
                self.ensemble_model, X, y, 
                cv=Config.CV_FOLDS, 
                scoring='neg_root_mean_squared_error', 
                n_jobs=-1
            )
            
            mae_scores = cross_val_score(
                self.ensemble_model, X, y, 
                cv=Config.CV_FOLDS, 
                scoring='neg_mean_absolute_error', 
                n_jobs=-1
            )
            
            self.model_metrics = {
                'EnsembleModel': {
                    'test_r2': np.mean(r2_scores),
                    'test_r2_std': np.std(r2_scores),
                    'test_rmse': np.mean(-rmse_scores),
                    'test_rmse_std': np.std(-rmse_scores),
                    'test_mae': np.mean(-mae_scores),
                    'test_mae_std': np.std(-mae_scores),
                }
            }
            
            self.logger.info(f"Ensemble Performance - R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
            self.logger.info(f"Ensemble Performance - RMSE: {np.mean(-rmse_scores):.2f} (±{np.std(-rmse_scores):.2f})")
            
        except Exception as e:
            self.logger.warning(f"Performance evaluation failed: {str(e)}")
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        try:
            predictions = self.ensemble_model.predict(X)
            return np.maximum(0, predictions)  # Ensure non-negative predictions
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {str(e)}")
            raise
    
    def save(self, filepath=None):
        """Save ensemble model"""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(Paths.MODEL_SAVE_PATH), 'ensemble_model.pkl')
        
        try:
            model_data = {
                'ensemble_model': self.ensemble_model,
                'preprocessor': self.preprocessor,
                'feature_names': self.feature_names,
                'model_metrics': self.model_metrics,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Ensemble model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save ensemble model: {str(e)}")
            return False
    
    def load(self, filepath=None):
        """Load ensemble model"""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(Paths.MODEL_SAVE_PATH), 'ensemble_model.pkl')
        
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Ensemble model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.ensemble_model = model_data['ensemble_model']
            self.preprocessor = model_data.get('preprocessor')
            self.feature_names = model_data.get('feature_names')
            self.model_metrics = model_data.get('model_metrics', {})
            self.is_trained = model_data.get('is_trained', False)
            
            self.logger.info(f"Ensemble model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble model: {str(e)}")
            return False
    
    def get_model_performance(self):
        """Get model performance metrics"""
        return self.model_metrics
