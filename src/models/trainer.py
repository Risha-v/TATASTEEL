from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from config.config import Config
from config.paths import Paths
import joblib, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, preprocessor):
        # preprocessor is a Pipeline: ('fe', FeatureEngineer), ('ct', ColumnTransformer)
        self.preprocessor = preprocessor
        self.best_model = None
        self.best_model_name = ""
        self.feature_names = None
        self.model_metrics = {}
        os.makedirs(Config.CATBOOST_TRAIN_DIR, exist_ok=True)

    def _build(self, model):
        # Single, consistent pipeline for all models keeps feature names intact
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])

    def train_and_select_model(self, X, y):
        try:
            models = {
                "CatBoost": CatBoostRegressor(**Config.CATBOOST_PARAMS),
                "XGBoost": XGBRegressor(**Config.XGBOOST_PARAMS),
                "LightGBM": LGBMRegressor(**Config.LIGHTGBM_PARAMS),
                "RandomForest": RandomForestRegressor(**Config.RANDOM_FOREST_PARAMS),
                "DecisionTree": None
            }

            logger.info("Starting model training and evaluation...")
            cv = KFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

            # Evaluate
            for name, model in models.items():
                if model is None:
                    continue
                logger.info(f"Training {name}...")
                try:
                    pipe = self._build(model)
                    cv_results = cross_validate(
                        pipe, X, y,
                        cv=cv,
                        scoring=Config.SCORING_METRICS,
                        n_jobs=1,
                        return_train_score=True,
                        error_score='raise'
                    )
                    self.model_metrics[name] = {
                        'train_r2': float(np.mean(cv_results['train_r2'])),
                        'test_r2': float(np.mean(cv_results['test_r2'])),
                        'test_rmse': float(np.mean(-cv_results['test_neg_root_mean_squared_error'])),
                        'test_mae': float(np.mean(-cv_results['test_neg_mean_absolute_error']))
                    }
                    logger.info(f"{name} OK - R2: {self.model_metrics[name]['test_r2']:.4f}")
                except Exception as e:
                    logger.error(f"{name} FAILED: {str(e)}")
                    self.model_metrics[name] = {
                        'train_r2': 0.0,
                        'test_r2': 0.0,
                        'test_rmse': 9e9,
                        'test_mae': 9e9
                    }

            # Select winner
            self.best_model_name = max(self.model_metrics, key=lambda m: self.model_metrics[m]['test_r2'])
            best_model = models[self.best_model_name]
            self.best_model = self._build(best_model)
            self.best_model.fit(X, y)
            logger.info(f"Selected best model: {self.best_model_name}")
            logger.info(f"Best model R2 score: {self.model_metrics[self.best_model_name]['test_r2']:.4f}")
            return self.best_model
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
        finally:
            self._cleanup_catboost_files()

    def _cleanup_catboost_files(self):
        try:
            import shutil
            if os.path.exists(Config.CATBOOST_TRAIN_DIR):
                shutil.rmtree(Config.CATBOOST_TRAIN_DIR, ignore_errors=True)
            logger.info("Cleaned up CatBoost temporary files")
        except Exception as e:
            logger.warning(f"Could not clean CatBoost files: {str(e)}")

    def generate_reports(self):
        try:
            os.makedirs(Paths.REPORTS_DIR, exist_ok=True)
            metrics_df = pd.DataFrame(self.model_metrics).T
            metrics_df.to_csv(os.path.join(Paths.REPORTS_DIR, 'model_performance.csv'))
            logger.info("Model performance metrics saved")
            self._generate_model_comparison_plot(metrics_df)
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")

    def _generate_model_comparison_plot(self, metrics_df):
        try:
            plot_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
            plt.figure(figsize=(14, 8))
            plt.subplot(1, 2, 1)
            sns.barplot(x='Model', y='test_r2', data=plot_df)
            plt.title('Model Comparison - R²')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.subplot(1, 2, 2)
            sns.barplot(x='Model', y='test_rmse', data=plot_df)
            plt.title('Model Comparison - RMSE')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(Paths.REPORTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Model comparison plot generated")
        except Exception as e:
            logger.error(f"Error generating model comparison plot: {str(e)}")

    def save_model(self):
        try:
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_names': None,
                'metrics': self.model_metrics,
                'config': {
                    'numerical_features': Config.NUMERICAL_FEATURES,
                    'categorical_features': Config.CATEGORICAL_FEATURES,
                    'target': Config.TARGET
                }
            }
            os.makedirs(os.path.dirname(Paths.MODEL_SAVE_PATH), exist_ok=True)
            joblib.dump(model_data, Paths.MODEL_SAVE_PATH)
            logger.info(f"Model saved to {Paths.MODEL_SAVE_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
