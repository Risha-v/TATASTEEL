import os
import tempfile

class Config:
    CATBOOST_TRAIN_DIR = os.path.join(tempfile.gettempdir(), "catboost_work_dir")

    RANDOM_FOREST_PARAMS = {
        'n_estimators': 600,
        'max_depth': 16,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'bootstrap': True
    }

    XGBOOST_PARAMS = {
        'n_estimators': 1500,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'min_child_weight': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'rmse',
        'tree_method': 'hist'
    }

    LIGHTGBM_PARAMS = {
        'n_estimators': 1200,
        'max_depth': -1,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'min_child_samples': 20,
        'num_leaves': 64,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'boosting_type': 'gbdt'
    }

    # Clean CatBoost params: remove incompatible options; Bayesian bootstrap without subsample/rsm
    CATBOOST_PARAMS = {
        'iterations': 2500,
        'depth': 6,
        'learning_rate': 0.03,
        'random_seed': 42,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'silent': True,
        'train_dir': CATBOOST_TRAIN_DIR,
        'allow_writing_files': False,
        'l2_leaf_reg': 3.0,
        'leaf_estimation_iterations': 10,
        'eval_metric': 'RMSE',
        'task_type': 'CPU',
        'thread_count': -1
    }

    DECISION_TREE_PARAMS = {
        'max_depth': 12,
        'min_samples_split': 8,
        'min_samples_leaf': 4,
        'random_state': 42,
        'criterion': 'squared_error',
        'splitter': 'best'
    }

    TEST_SIZE = 0.2
    CV_FOLDS = 5
    RANDOM_STATE = 42
    OUTLIER_THRESHOLD = 2.5

    SCORING_METRICS = [
        'r2',
        'neg_mean_squared_error',
        'neg_root_mean_squared_error',
        'neg_mean_absolute_error'
    ]

    NUMERICAL_FEATURES = ['age', 'bmi', 'children']
    CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']
    TARGET = 'charges'

    VALIDATION_RULES = {
        'age': (18, 100),
        'bmi': (15.0, 50.0),
        'children': (0, 10),
        'sex': ['male', 'female'],
        'smoker': ['yes', 'no'],
        'region': ['northeast', 'northwest', 'southeast', 'southwest']
    }
