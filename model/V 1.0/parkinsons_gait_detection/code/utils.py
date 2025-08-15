"""
Utility functions for Parkinson's Disease Detection Pipeline
===========================================================

This module contains helper functions for data processing, evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix, roc_curve, auc # type: ignore
import joblib
from pathlib import Path

def load_model(model_path):
    """
    Load a saved model package

    Args:
        model_path (str): Path to saved model file

    Returns:
        dict: Model package with all components
    """
    try:
        model_package = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"Model type: {model_package['model_name']}")
        print(f"Features: {len(model_package['selected_features'])}")
        return model_package
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_new_data(model_package, new_data):
    """
    Make predictions on new data using a saved model

    Args:
        model_package (dict): Loaded model package
        new_data (pd.DataFrame): New data to predict

    Returns:
        tuple: Predictions and probabilities
    """
    model = model_package['model']
    scaler = model_package['scaler']
    feature_selector = model_package['feature_selector']
    selected_features = model_package['selected_features']

    # Select only the features used during training
    available_features = [f for f in selected_features if f in new_data.columns]
    missing_features = [f for f in selected_features if f not in new_data.columns]

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        return None, None

    X_new = new_data[available_features]

    # Apply same preprocessing
    if scaler:
        X_new_scaled = scaler.transform(X_new)
    else:
        X_new_scaled = X_new.values

    if feature_selector:
        X_new_selected = feature_selector.transform(X_new_scaled)
    else:
        X_new_selected = X_new_scaled

    # Make predictions
    predictions = model.predict(X_new_selected)
    probabilities = model.predict_proba(X_new_selected) if hasattr(model, 'predict_proba') else None

    return predictions, probabilities

def plot_confusion_matrix(y_true, y_pred, labels=['Control', "Parkinson's"], title='Confusion Matrix'):
    """
    Plot confusion matrix with annotations

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return cm

def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    return roc_auc

def validate_data_format(df, required_columns=None):
    """
    Validate data format and structure

    Args:
        df (pd.DataFrame): Data to validate
        required_columns (list): List of required column names

    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'issues': [],
        'warnings': []
    }

    # Check basic structure
    if df.empty:
        results['valid'] = False
        results['issues'].append("Dataset is empty")
        return results

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        results['warnings'].append(f"Missing values found in {missing[missing > 0].index.tolist()}")

    # Check data types
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if 'label' in non_numeric:
        non_numeric.remove('label')
    if non_numeric:
        results['warnings'].append(f"Non-numeric columns found: {non_numeric}")

    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results['valid'] = False
            results['issues'].append(f"Missing required columns: {missing_cols}")

    # Check for outliers (simple z-score based)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'label':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > 3).sum()
            if outliers > 0:
                results['warnings'].append(f"{col}: {outliers} potential outliers (z-score > 3)")

    return results

def generate_sample_data(n_control=50, n_patients=60, save_path=None):
    """
    Generate sample gait data for testing purposes

    Args:
        n_control (int): Number of control subjects
        n_patients (int): Number of patient subjects
        save_path (str): Path to save the data

    Returns:
        tuple: (control_data, patients_data)
    """
    np.random.seed(42)

    # Control group data (better gait parameters)
    control_data = pd.DataFrame({
        'subject_id': [f'C{i:03d}' for i in range(1, n_control+1)],
        'age': np.random.normal(65, 8, n_control),
        'gender': np.random.choice(['M', 'F'], n_control),
        'stride_length_left': np.random.normal(1.3, 0.1, n_control),
        'stride_length_right': np.random.normal(1.3, 0.1, n_control),
        'gait_speed': np.random.normal(1.2, 0.15, n_control),
        'cadence': np.random.normal(115, 8, n_control),
        'stride_time': np.random.normal(1.05, 0.05, n_control),
        'swing_time_left': np.random.normal(0.42, 0.03, n_control),
        'swing_time_right': np.random.normal(0.42, 0.03, n_control),
        'stance_time_left': np.random.normal(0.63, 0.04, n_control),
        'stance_time_right': np.random.normal(0.63, 0.04, n_control),
        'double_support_time': np.random.normal(0.12, 0.015, n_control),
        'step_width': np.random.normal(0.12, 0.02, n_control),
        'stride_length_variability': np.random.normal(0.03, 0.008, n_control),
        'stride_time_variability': np.random.normal(0.025, 0.005, n_control),
        'arm_swing_amplitude_left': np.random.normal(25, 4, n_control),
        'arm_swing_amplitude_right': np.random.normal(25, 4, n_control)
    })

    # Patient data (impaired gait parameters)
    patients_data = pd.DataFrame({
        'subject_id': [f'P{i:03d}' for i in range(1, n_patients+1)],
        'age': np.random.normal(68, 9, n_patients),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'stride_length_left': np.random.normal(1.0, 0.15, n_patients),
        'stride_length_right': np.random.normal(1.0, 0.15, n_patients),
        'gait_speed': np.random.normal(0.9, 0.2, n_patients),
        'cadence': np.random.normal(105, 12, n_patients),
        'stride_time': np.random.normal(1.2, 0.1, n_patients),
        'swing_time_left': np.random.normal(0.38, 0.05, n_patients),
        'swing_time_right': np.random.normal(0.38, 0.05, n_patients),
        'stance_time_left': np.random.normal(0.72, 0.06, n_patients),
        'stance_time_right': np.random.normal(0.72, 0.06, n_patients),
        'double_support_time': np.random.normal(0.16, 0.03, n_patients),
        'step_width': np.random.normal(0.15, 0.03, n_patients),
        'stride_length_variability': np.random.normal(0.08, 0.02, n_patients),
        'stride_time_variability': np.random.normal(0.06, 0.015, n_patients),
        'arm_swing_amplitude_left': np.random.normal(15, 6, n_patients),
        'arm_swing_amplitude_right': np.random.normal(15, 6, n_patients)
    })

    # Ensure positive values where appropriate
    positive_cols = ['stride_length_left', 'stride_length_right', 'gait_speed', 
                    'cadence', 'stride_time', 'swing_time_left', 'swing_time_right',
                    'stance_time_left', 'stance_time_right', 'double_support_time',
                    'step_width', 'stride_length_variability', 'stride_time_variability']

    for col in positive_cols:
        control_data[col] = np.abs(control_data[col])
        patients_data[col] = np.abs(patients_data[col])

    if save_path:
        control_path = f"{save_path}/control_group_selected_variables.csv"
        patients_path = f"{save_path}/patients_selected_variables.csv"

        control_data.to_csv(control_path, index=False)
        patients_data.to_csv(patients_path, index=False)

        print(f"Sample data saved:")
        print(f"  Control group: {control_path}")
        print(f"  Patients group: {patients_path}")

    return control_data, patients_data

def create_feature_documentation():
    """
    Generate documentation for gait features

    Returns:
        dict: Feature descriptions and clinical significance
    """
    feature_docs = {
        'spatiotemporal': {
            'stride_length': 'Distance covered in one complete gait cycle (heel strike to heel strike)',
            'step_length': 'Distance from heel strike of one foot to heel strike of opposite foot',
            'gait_speed': 'Average walking velocity (m/s)',
            'cadence': 'Number of steps per minute',
            'step_width': 'Lateral distance between feet during walking'
        },
        'temporal': {
            'stride_time': 'Time for one complete gait cycle',
            'step_time': 'Time from one heel strike to opposite heel strike', 
            'swing_time': 'Time when foot is off the ground during gait cycle',
            'stance_time': 'Time when foot is in contact with ground',
            'double_support_time': 'Time when both feet are in contact with ground'
        },
        'variability': {
            'stride_length_variability': 'Coefficient of variation in stride length',
            'stride_time_variability': 'Coefficient of variation in stride time',
            'step_length_variability': 'Coefficient of variation in step length',
            'gait_speed_variability': 'Coefficient of variation in walking speed'
        },
        'kinematic': {
            'arm_swing_amplitude': 'Angular displacement of arm during walking',
            'hip_flexion_angle': 'Maximum hip flexion during gait cycle',
            'knee_flexion_angle': 'Maximum knee flexion during gait cycle',
            'ankle_dorsiflexion': 'Maximum ankle dorsiflexion angle'
        },
        'clinical_significance': {
            'parkinson_effects': {
                'reduced_gait_speed': 'Bradykinesia (slowness of movement)',
                'shorter_stride_length': 'Hypokinesia (reduced amplitude)',
                'increased_variability': 'Motor control deficits',
                'prolonged_double_support': 'Compensatory balance strategy',
                'reduced_arm_swing': 'Rigidity and coordination issues'
            }
        }
    }

    return feature_docs

def export_results_summary(detector, output_path='results_summary.csv'):
    """
    Export detailed results summary

    Args:
        detector: ParkinsonDetector instance with results
        output_path: Path to save the summary
    """
    if not detector.results:
        print("No results to export")
        return

    summary_data = []
    for model_name, results in detector.results.items():
        summary_data.append({
            'Model': model_name,
            'Test_Accuracy': results['accuracy'],
            'AUC_Score': results.get('auc', 'N/A'),
            'CV_Mean': results['cv_mean'],
            'CV_Std': results['cv_std'],
            'CV_Range': f"{results['cv_mean'] - results['cv_std']:.3f} - {results['cv_mean'] + results['cv_std']:.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"Results summary exported to: {output_path}")

    return summary_df

# Example usage functions
def quick_analysis(control_path, patients_path):
    """
    Run a quick analysis with default settings
    """
    from parkinsons_detector import ParkinsonDetector

    detector = ParkinsonDetector()
    results = detector.run_complete_pipeline(control_path, patients_path)

    return detector, results

if __name__ == "__main__":
    # Generate sample data for testing
    print("Generating sample data...")
    control_data, patients_data = generate_sample_data(save_path='../data')

    print("\nFeature documentation:")
    docs = create_feature_documentation()
    for category, features in docs.items():
        if category != 'clinical_significance':
            print(f"\n{category.upper()}:")
            for feature, description in features.items():
                print(f"  {feature}: {description}")
