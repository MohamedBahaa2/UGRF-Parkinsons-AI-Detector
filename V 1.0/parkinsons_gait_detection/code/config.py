"""
Configuration file for Parkinson's Disease Detection Pipeline
============================================================
"""

# Data processing configuration
DATA_CONFIG = {
    'test_size': 0.2,           # Proportion of data for testing
    'random_state': 42,         # Random seed for reproducibility
    'cv_folds': 5,              # Number of cross-validation folds
    'stratify': True,           # Use stratified sampling
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'scale_features': True,     # Apply feature scaling
    'feature_selection': True,  # Enable feature selection
    'n_features': 10,          # Number of features to select
    'selection_method': 'univariate',  # 'univariate', 'rfe', or 'importance'
}

# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'svm_rbf': {
        'C': 10,
        'gamma': 'scale',
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42
    },
    'svm_linear': {
        'C': 1,
        'kernel': 'linear',
        'probability': True,
        'random_state': 42
    },
    'logistic_regression': {
        'C': 1,
        'max_iter': 1000,
        'random_state': 42
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'uniform'
    }
}

# Evaluation configuration
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'cv_scoring': 'accuracy',
    'save_results': True,
    'plot_results': True
}

# File paths
PATHS = {
    'data_dir': 'data/',
    'results_dir': 'results/',
    'models_dir': 'results/models/',
    'plots_dir': 'results/plots/'
}

# Expected column names (update based on your data)
EXPECTED_COLUMNS = [
    'stride_length',
    'gait_speed', 
    'cadence',
    'stride_time',
    'swing_time',
    'stance_time',
    'double_support_time',
    'stride_length_variability',
    'step_width',
    'arm_swing_amplitude'
]

# Columns to exclude (as requested by user)
EXCLUDE_COLUMNS = [
    'UPDRS_III', 'HY', 'UPDRS-III', 'H&Y', 
    'updrs', 'hy', 'updrs_iii', 'hoehn_yahr'
]
