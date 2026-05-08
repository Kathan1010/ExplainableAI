import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_models():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_data.csv')
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Map target to integers for XGBoost
    target_mapping = {'Fail': 0, 'At-Risk': 1, 'Pass': 2}
    y_mapped = y.map(target_mapping)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)
    
    numeric_features = ['num_courses_enrolled', 'study_hours_per_week', 'sleep_hours_avg', 
                        'extracurricular_count', 'prev_gpa', 'assignment_submission_rate', 
                        'midterm_score', 'financial_stress_index']
    categorical_features = ['gender', 'region', 'parent_education_level', 'internet_access_quality']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])
    
    # Fit preprocessor on X_train and transform X_test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    
    # Define models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        trained_models[name] = model
        
    # Save models, preprocessor, and feature names
    output_dir = os.path.dirname(__file__)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    joblib.dump(trained_models, os.path.join(output_dir, 'models.joblib'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.joblib'))
    
    # Save the test set for evaluation later
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

if __name__ == "__main__":
    train_models()
