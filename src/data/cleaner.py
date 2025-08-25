import pandas as pd
import numpy as np
from scipy import stats
from config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataCleaner:
    def __init__(self, data):
        self.data = data.copy()
        self.cleaned_data = None

    def clean_data(self):
        """Run complete data cleaning pipeline"""
        try:
            self.cleaned_data = self.data.copy()
            original_size = len(self.cleaned_data)

            # Remove null values
            self._handle_missing_values()

            # Handle outliers
            self._handle_outliers()

            # Remove logical inconsistencies
            self._remove_logical_inconsistencies()

            # Remove unrealistic values
            self._remove_unrealistic_values()

            cleaned_size = len(self.cleaned_data)
            logger.info(f"Data cleaned. Original size: {original_size}, Cleaned size: {cleaned_size}")

            return self.cleaned_data

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def _handle_missing_values(self):
        """Handle missing values"""
        initial_size = len(self.cleaned_data)

        # Remove rows with any missing values
        self.cleaned_data = self.cleaned_data.dropna()
        removed = initial_size - len(self.cleaned_data)

        if removed > 0:
            logger.info(f"Removed {removed} rows with missing values")

    def _handle_outliers(self):
        """Handle outliers using robust scaling and capping"""
        # Handle charges outliers
        charges = self.cleaned_data[Config.TARGET]
        q1 = charges.quantile(0.25)
        q3 = charges.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        self.cleaned_data[Config.TARGET] = self.cleaned_data[Config.TARGET].clip(
            lower_bound, upper_bound
        )

        # Handle BMI outliers
        bmi = self.cleaned_data['bmi']
        q1_bmi = bmi.quantile(0.25)
        q3_bmi = bmi.quantile(0.75)
        iqr_bmi = q3_bmi - q1_bmi
        lower_bound_bmi = q1_bmi - 1.5 * iqr_bmi
        upper_bound_bmi = q3_bmi + 1.5 * iqr_bmi

        self.cleaned_data['bmi'] = self.cleaned_data['bmi'].clip(
            lower_bound_bmi, upper_bound_bmi
        )

        logger.info("Handled outliers in charges and BMI")

    def _remove_logical_inconsistencies(self):
        """Remove logically inconsistent records"""
        initial_size = len(self.cleaned_data)

        # Find median charges for non-smokers by demographics
        nonsmoker_data = self.cleaned_data[self.cleaned_data['smoker'] == 'no']
        
        if len(nonsmoker_data) > 0:
            nonsmoker_medians = nonsmoker_data.groupby(['sex', 'region'])[Config.TARGET].median()

            # Remove smokers with charges lower than non-smokers in same demographic
            for (gender, region), median_charge in nonsmoker_medians.items():
                if pd.isna(median_charge):
                    continue

                inconsistent_mask = (
                    (self.cleaned_data['sex'] == gender) &
                    (self.cleaned_data['region'] == region) &
                    (self.cleaned_data['smoker'] == 'yes') &
                    (self.cleaned_data[Config.TARGET] < median_charge * 0.8)
                )

                inconsistent_count = inconsistent_mask.sum()
                if inconsistent_count > 0:
                    self.cleaned_data = self.cleaned_data[~inconsistent_mask]
                    logger.info(f"Removed {inconsistent_count} inconsistent records for {gender} in {region}")

        removed = initial_size - len(self.cleaned_data)
        if removed > 0:
            logger.info(f"Total inconsistent records removed: {removed}")

    def _remove_unrealistic_values(self):
        """Remove records with unrealistic values"""
        initial_size = len(self.cleaned_data)

        # Apply validation rules
        valid_mask = (
            (self.cleaned_data['age'] >= Config.VALIDATION_RULES['age'][0]) &
            (self.cleaned_data['age'] <= Config.VALIDATION_RULES['age'][1]) &
            (self.cleaned_data['bmi'] >= Config.VALIDATION_RULES['bmi'][0]) &
            (self.cleaned_data['bmi'] <= Config.VALIDATION_RULES['bmi'][1]) &
            (self.cleaned_data['children'] >= Config.VALIDATION_RULES['children'][0]) &
            (self.cleaned_data['children'] <= Config.VALIDATION_RULES['children'][1]) &
            (self.cleaned_data['sex'].isin(Config.VALIDATION_RULES['sex'])) &
            (self.cleaned_data['smoker'].isin(Config.VALIDATION_RULES['smoker'])) &
            (self.cleaned_data['region'].isin(Config.VALIDATION_RULES['region'])) &
            (self.cleaned_data[Config.TARGET] > 0)
        )

        self.cleaned_data = self.cleaned_data[valid_mask]
        removed = initial_size - len(self.cleaned_data)

        if removed > 0:
            logger.info(f"Removed {removed} records with unrealistic values")
