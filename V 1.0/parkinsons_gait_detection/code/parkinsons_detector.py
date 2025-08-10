#!/usr/bin/env python3
"""
Parkinson's Disease Detection using Gait Analysis Data
=====================================================

This script provides a complete machine learning pipeline for detecting
Parkinson's disease using gait analysis parameters.

Author: AI Research Assistant
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, roc_curve)

class ParkinsonDetector:
    """
    Complete pipeline for Parkinson's Disease detection using gait data
    """

    def __init__(self, config=None):
        """Initialize the detector with configuration"""
        self.config = config or self._default_config()
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.results = {}

    def _default_config(self):
        """Default configuration parameters"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'n_features': 10,
            'feature_selection': True,
            'scale_features': True
        }

    def load_and_prepare_data(self, control_path, patients_path):
        """
        Load control and patient data, combine them, and create labels

        Args:
            control_path (str): Path to control group CSV file
            patients_path (str): Path to patients CSV file

        Returns:
            pd.DataFrame: Combined dataset with labels
        """
        try:
            # Load datasets
            print("Loading data files...")
            control_data = pd.read_csv(control_path)
            patients_data = pd.read_csv(patients_path)

            print(f"Control group: {len(control_data)} subjects")
            print(f"Patient group: {len(patients_data)} subjects")

            # Add labels
            control_data['label'] = 0  # Control group
            patients_data['label'] = 1  # Parkinson's patients

            # Combine datasets
            combined_data = pd.concat([control_data, patients_data], ignore_index=True)

            # Remove UPDRS-III and H&Y columns if present
            columns_to_remove = ['UPDRS_III', 'HY', 'UPDRS-III', 'H&Y', 'updrs', 'hy']
            for col in columns_to_remove:
                if col in combined_data.columns:
                    combined_data = combined_data.drop(col, axis=1)
                    print(f"Removed column: {col}")

            print(f"Final dataset shape: {combined_data.shape}")
            return combined_data

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def explore_data(self, df):
        """
        Perform exploratory data analysis

        Args:
            df (pd.DataFrame): Dataset to explore
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)

        print(f"Dataset shape: {df.shape}")
        print(f"Control subjects: {len(df[df['label']==0])}")
        print(f"Parkinson's subjects: {len(df[df['label']==1])}")

        print("\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found")
        else:
            print(missing[missing > 0])

        print("\nBasic statistics:")
        print(df.describe())

        # Save exploration results
        df.describe().to_csv('results/data_statistics.csv')

        return df


    def preprocess_data(self, df):
        """
        Preprocess the data: handling missing values, scaling, feature selection, train-test split
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Handle missing values FIRST
        print("Handling missing values...")
        
        # Fill disease_duration_years for control group (they don't have PD)
        if 'disease_duration_years' in df.columns:
            df.loc[df['label'] == 0, 'disease_duration_years'] = 0
            print("Filled disease_duration_years for control group")
        
        # Check for any remaining missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Remaining missing values:")
            print(missing_counts[missing_counts > 0])
            
            # Remove columns with too many missing values (>50%)
            threshold = len(df) * 0.5
            cols_to_drop = missing_counts[missing_counts > threshold].index
            if len(cols_to_drop) > 0:
                df = df.drop(cols_to_drop, axis=1)
                print(f"Dropped columns with >50% missing: {list(cols_to_drop)}")
            
            # Use median imputation for remaining missing values
            from sklearn.impute import SimpleImputer
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_label_cols = [col for col in numeric_cols if col != 'label']
            
            if len(non_label_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                df[non_label_cols] = imputer.fit_transform(df[non_label_cols])
                print(" Applied median imputation to remaining missing values")
        
        # Separate features (X) and target (y)
        X = df.drop('label', axis=1).select_dtypes(include=[np.number])
        y = df['label']

            # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Scale features if configured
        if self.config['scale_features']:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            print("Features scaled using StandardScaler")
        
        # Feature selection if configured
        if self.config['feature_selection'] and self.config['n_features'] < X.shape[1]:
            self.feature_selector = SelectKBest(score_func=f_classif, k=self.config['n_features'])
            X_train = self.feature_selector.fit_transform(X_train, y_train)
            X_test = self.feature_selector.transform(X_test)
            self.selected_features = X.columns[self.feature_selector.get_support()]
            print(f"Selected {len(self.selected_features)} features: {list(self.selected_features)}")
        else:
            self.selected_features = X.columns
        
        # ✅ Return final training and test sets
        return X_train, X_test, y_train, y_test
    


    '''def preprocess_data(self, df):
        """
        Preprocess the data: scaling, feature selection, train-test split

        Args:
            df (pd.DataFrame): Input dataset

        Returns:
            tuple: Preprocessed training and test sets
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)

        # Separate features and target
        X = df.drop('label', axis=1)
        y = df['label']

        # Handle non-numeric columns
        X = X.select_dtypes(include=[np.number])

        print(f"Features: {list(X.columns)}")
        print(f"Number of features: {X.shape[1]}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Scale features if configured
        if self.config['scale_features']:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            print("Features scaled using StandardScaler")
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Feature selection if configured
        if self.config['feature_selection'] and self.config['n_features'] < X.shape[1]:
            self.feature_selector = SelectKBest(
                score_func=f_classif, 
                k=self.config['n_features']
            )
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            self.selected_features = X.columns[self.feature_selector.get_support()]
            print(f"Selected {len(self.selected_features)} features:")
            for i, feature in enumerate(self.selected_features, 1):
                print(f"  {i}. {feature}")
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            self.selected_features = X.columns

        return X_train_selected, X_test_selected, y_train, y_test'''

    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models and evaluate performance

        Args:
            X_train, X_test, y_train, y_test: Preprocessed datasets
        """
        print("\n" + "="*60)
        print("MODEL TRAINING AND EVALUATION")
        print("="*60)

        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.config['random_state']
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', 
                C=10, 
                probability=True, 
                random_state=self.config['random_state']
            ),
            'SVM (Linear)': SVC(
                kernel='linear', 
                probability=True, 
                random_state=self.config['random_state']
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.config['random_state'], 
                max_iter=1000
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

            # Cross-validation
            cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, 
                               random_state=self.config['random_state'])
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            # Print results
            print(f"  Test Accuracy: {accuracy:.3f}")
            if auc:
                print(f"  AUC Score: {auc:.3f}")
            print(f"  CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

        return self.results

    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            return None, None

        best_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_name]['model']

        return best_name, best_model

    def detailed_evaluation(self, y_test):
        """
        Provide detailed evaluation of the best model

        Args:
            y_test: True test labels
        """
        print("\n" + "="*60)
        print("DETAILED MODEL EVALUATION")
        print("="*60)

        best_name, best_model = self.get_best_model()
        if best_model is None:
            print("No models trained yet!")
            return

        best_results = self.results[best_name]
        y_pred = best_results['y_pred']

        print(f"Best Model: {best_name}")
        print(f"Test Accuracy: {best_results['accuracy']:.3f}")
        if best_results['auc']:
            print(f"AUC Score: {best_results['auc']:.3f}")

        print("\nClassification Report:")
        print("-" * 40)
        print(classification_report(y_test, y_pred, target_names=['Control', 'Parkinson\'s']))

        print("\nConfusion Matrix:")
        print("-" * 20)
        cm = confusion_matrix(y_test, y_pred)
        print("Predicted:    Control  Parkinson's")
        print(f"Control          {cm[0,0]:2d}        {cm[0,1]:2d}")
        print(f"Parkinson's      {cm[1,0]:2d}        {cm[1,1]:2d}")

        # Feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            print("\nFeature Importance:")
            print("-" * 25)
            importance = best_model.feature_importances_
            feature_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance': importance
            }).sort_values('importance', ascending=False)

            for _, row in feature_df.iterrows():
                print(f"{row['feature']:<25}: {row['importance']:.3f}")

            # Save feature importance
            feature_df.to_csv('results/feature_importance.csv', index=False)

    def save_model(self, filepath=None):
        """Save the best model and preprocessing components"""
        best_name, best_model = self.get_best_model()
        if best_model is None:
            print("No model to save!")
            return

        if filepath is None:
            filepath = f'results/best_model_{best_name.lower().replace(" ", "_")}.joblib'

        # Save model and preprocessing components
        model_package = {
            'model': best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': list(self.selected_features),
            'config': self.config,
            'model_name': best_name,
            'performance': self.results[best_name]
        }

        joblib.dump(model_package, filepath)
        print(f"Model saved to: {filepath}")

    def run_complete_pipeline(self, control_path, patients_path):
        """
        Run the complete machine learning pipeline

        Args:
            control_path (str): Path to control group data
            patients_path (str): Path to patient group data
        """
        print("PARKINSON'S DISEASE DETECTION PIPELINE")
        print("="*60)

        # Load and prepare data
        df = self.load_and_prepare_data(control_path, patients_path)
        if df is None:
            return

        # Explore data
        df = self.explore_data(df)

        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(df)

        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test)

        # Detailed evaluation
        self.detailed_evaluation(y_test)

        # Save best model
        self.save_model()

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the 'results/' folder for detailed outputs.")

        return results


def main():
    """Main execution function"""
    # Example usage
    detector = ParkinsonDetector()

    # Update these paths to your actual data files
    control_path = "data/control_group_selected_variables.csv"
    patients_path = "data/patients_selected_variables.csv"

    # Check if data files exist
    if not (Path(control_path).exists() and Path(patients_path).exists()):
        print("Data files not found!")
        print("Please place your CSV files in the 'data/' folder:")
        print(f"  - {control_path}")
        print(f"  - {patients_path}")
        return

    # Run complete pipeline
    results = detector.run_complete_pipeline(control_path, patients_path)

    return detector, results


if __name__ == "__main__":
    detector, results = main()
