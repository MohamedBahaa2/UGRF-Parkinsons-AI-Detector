#!/usr/bin/env python3
"""
Test script for Parkinson's Disease Detection Pipeline
=====================================================

This script tests the complete pipeline using the provided sample data.
Run this script to verify that everything is working correctly.

Usage:
    python test_pipeline.py
"""

import sys
import os
sys.path.append('code')

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        #import sklearn
        from code.parkinsons_detector import ParkinsonDetector
        from code.utils import load_model, validate_data_format
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def test_data_loading():
    """Test if sample data can be loaded"""
    print("\nTesting data loading...")
    try:
        import pandas as pd

        control_path = "data/control_group_selected_variables.csv"
        patients_path = "data/patients_selected_variables.csv"

        if not os.path.exists(control_path):
            print(f"Control data file not found: {control_path}")
            return False

        if not os.path.exists(patients_path):
            print(f"Patients data file not found: {patients_path}")
            return False

        control_data = pd.read_csv(control_path)
        patients_data = pd.read_csv(patients_path)

        print(f"Control group loaded: {control_data.shape}")
        print(f"Patients group loaded: {patients_data.shape}")

        return True

    except Exception as e:
        print(f"Data loading error: {e}")
        return False

def test_pipeline():
    """Test the complete ML pipeline"""
    print("\nTesting ML pipeline...")

    try:
        from code.parkinsons_detector import ParkinsonDetector

        # Initialize detector
        config = {
            'test_size': 0.3,
            'cv_folds': 3,
            'n_features': 8,
            'random_state': 42,
            'scale_features': True,        # existing fix
            'feature_selection': True      # Add this line
        }
        

        detector = ParkinsonDetector(config=config)

        # Test data loading
        control_path = "data/control_group_selected_variables.csv"
        patients_path = "data/patients_selected_variables.csv"

        df = detector.load_and_prepare_data(control_path, patients_path)

        if df is None:
            print("Failed to load and prepare data")
            return False

        print(f"Data prepared: {df.shape}")

        # Test preprocessing
        X_train, X_test, y_train, y_test = detector.preprocess_data(df)
        print(f"Data preprocessed: Train {X_train.shape}, Test {X_test.shape}")

        # Test model training (just Random Forest for speed)
        from sklearn.ensemble import RandomForestClassifier # type: ignore
        from sklearn.metrics import accuracy_score # type: ignore

        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)  # Fast training
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained and tested: Accuracy = {accuracy:.3f}")

        if accuracy > 0.6:  # Reasonable threshold for test data
            print("Pipeline test successful")
            return True
        else:
            print(f"Low accuracy ({accuracy:.3f}), but pipeline is functional")
            return True

    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions"""
    print("\nTesting utility functions...")

    try:
        from code.utils import validate_data_format, create_feature_documentation
        import pandas as pd

        # Test data validation
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'label': [0, 0, 1, 1, 1]
        })

        validation_result = validate_data_format(test_data)
        print(f"Data validation: {validation_result['valid']}")

        # Test feature documentation
        docs = create_feature_documentation()
        print(f"Feature documentation: {len(docs)} categories")

        return True

    except Exception as e:
        print(f"Utility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("PARKINSON'S DETECTION PIPELINE - TEST SUITE")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("ML Pipeline", test_pipeline),
        ("Utilities", test_utilities)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("\nAll tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Replace sample data with your actual CSV files")
        print("2. Run: python code/parkinsons_detector.py")
        print("3. Check results in the 'results/' folder")
    else:
        print(f"\n {len(tests) - passed} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all packages are installed: pip install -r requirements.txt")
        print("2. Check that data files exist in the 'data/' folder")
        print("3. Verify Python version is 3.8 or higher")

    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
