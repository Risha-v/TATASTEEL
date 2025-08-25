from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from config.config import Config
from src.data.feature_engineer import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.pipeline = None

    def create_preprocessor(self):
        try:
            # Feature engineer (sklearn transformer)
            fe = FeatureEngineer(enable=True)

            # Columns after FE
            num_feats = sorted(list(set(
                Config.NUMERICAL_FEATURES + ['age_bmi','age2','bmi2','smoker_bmi','region_factor']
            )))
            cat_feats = sorted(list(set(
                Config.CATEGORICAL_FEATURES + ['bmi_under','bmi_normal','bmi_over','bmi_obese']
            )))

            ct = ColumnTransformer(
                transformers=[
                    ('num', RobustScaler(), num_feats),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_feats)
                ],
                remainder='drop'
            )

            # Full preprocessing pipeline: FE -> ColumnTransformer
            self.pipeline = Pipeline([
                ('fe', fe),
                ('ct', ct)
            ])

            logger.info("Preprocessor created successfully")
            return self.pipeline
        except Exception as e:
            logger.error(f"Error creating preprocessor: {str(e)}")
            raise
