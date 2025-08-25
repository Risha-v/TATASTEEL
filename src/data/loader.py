import pandas as pd
import os
from config.paths import Paths
from config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataLoader:
    def __init__(self):
        self.data = None
        self.regional_charge_factors = None

    def load_data(self):
        """Load and validate the dataset"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(Paths.RAW_DATA), exist_ok=True)
            
            if not os.path.exists(Paths.RAW_DATA):
                raise FileNotFoundError(f"Data file not found at {Paths.RAW_DATA}")

            self.data = pd.read_csv(Paths.RAW_DATA)
            logger.info(f"Dataset loaded successfully with shape {self.data.shape}")
            
            self._validate_data_structure()
            self._calculate_regional_factors()
            
            return self.data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data_structure(self):
        """Validate the basic structure of the dataset"""
        required_columns = Config.NUMERICAL_FEATURES + Config.CATEGORICAL_FEATURES + [Config.TARGET]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check data types
        for col in Config.NUMERICAL_FEATURES + [Config.TARGET]:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(f"Column '{col}' should be numeric but is {self.data[col].dtype}")

        # Check for null values
        null_counts = self.data.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts[null_counts > 0].to_dict()}")

    def _calculate_regional_factors(self):
        """Calculate regional charge factors"""
        try:
            regional_means = self.data.groupby('region')[Config.TARGET].mean()
            overall_mean = self.data[Config.TARGET].mean()
            self.regional_charge_factors = (regional_means / overall_mean).to_dict()
            
            logger.info("Regional charge factors calculated")
            logger.info(f"Regional factors: {self.regional_charge_factors}")
            
        except Exception as e:
            logger.error(f"Error calculating regional factors: {str(e)}")
            self.regional_charge_factors = {}
