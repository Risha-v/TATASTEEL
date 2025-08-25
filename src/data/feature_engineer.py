import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature engineer.
    Learns region factors on fit(X, y) and produces deterministic new features on transform(X).
    """
    def __init__(self, enable=True):
        self.enable = enable
        self.region_factors_ = {}
        self.fitted_ = False

    def fit(self, X, y=None):
        if not self.enable:
            self.fitted_ = True
            return self
        # X is DataFrame. y is charges series when pipeline.fit is called with y.
        try:
            if y is not None and isinstance(X, pd.DataFrame):
                df = X.copy()
                df['_target_'] = y.values
                overall = df['_target_'].mean()
                reg = df.groupby('region')['_target_'].mean() / overall
                self.region_factors_ = reg.to_dict()
            else:
                self.region_factors_ = {}
        except Exception:
            self.region_factors_ = {}
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("FeatureEngineer not fitted")
        if not self.enable:
            return X
        out = X.copy()

        # BMI category one-hots
        bmi = out['bmi']
        out['bmi_under'] = (bmi < 18.5).astype(int)
        out['bmi_normal'] = ((bmi >= 18.5) & (bmi < 25)).astype(int)
        out['bmi_over'] = ((bmi >= 25) & (bmi < 30)).astype(int)
        out['bmi_obese'] = (bmi >= 30).astype(int)

        # Interactions and nonlinearity
        out['age_bmi'] = out['age'] * out['bmi']
        out['smoker_bmi'] = out['bmi'] * (out['smoker'].map({'yes':1,'no':0}).fillna(0))
        out['age2'] = out['age'] ** 2
        out['bmi2'] = out['bmi'] ** 2

        # Region factor from fit
        if self.region_factors_:
            out['region_factor'] = out['region'].map(self.region_factors_).fillna(1.0)
        else:
            out['region_factor'] = 1.0

        return out
