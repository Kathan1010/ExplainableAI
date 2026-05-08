import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_records=2000, random_seed=42):
    np.random.seed(random_seed)
    
    data = {}
    
    # 1. Socioeconomic and Demographic Features (Bias Features)
    data['gender'] = np.random.choice(['Male', 'Female', 'Non-binary'], p=[0.48, 0.48, 0.04], size=num_records)
    data['region'] = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.5, 0.3, 0.2], size=num_records)
    data['parent_education_level'] = np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], p=[0.4, 0.4, 0.15, 0.05], size=num_records)
    data['internet_access_quality'] = np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], p=[0.1, 0.3, 0.4, 0.2], size=num_records)
    data['financial_stress_index'] = np.random.randint(1, 11, size=num_records) # 1 to 10
    
    # 2. Traditional Academic and Behavioral Features
    data['num_courses_enrolled'] = np.random.randint(3, 7, size=num_records)
    data['study_hours_per_week'] = np.random.normal(loc=15, scale=5, size=num_records).clip(0, 40)
    data['sleep_hours_avg'] = np.random.normal(loc=7, scale=1.5, size=num_records).clip(3, 12)
    data['extracurricular_count'] = np.random.randint(0, 4, size=num_records)
    
    # Base GPA is influenced by study hours, sleep, and inverse of financial stress
    base_gpa = 2.0 + (data['study_hours_per_week'] / 40) * 1.5 + (data['sleep_hours_avg'] / 12) * 0.5 - (data['financial_stress_index'] / 10) * 0.5
    
    # Inject some bias: rural regions and high school parent education have slightly lower base GPA
    bias_mask_rural = (data['region'] == 'Rural')
    bias_mask_edu = (data['parent_education_level'] == 'High School')
    
    base_gpa[bias_mask_rural] -= 0.3
    base_gpa[bias_mask_edu] -= 0.2
    
    data['prev_gpa'] = np.random.normal(loc=base_gpa, scale=0.3).clip(1.0, 4.0)
    
    # Midterm and assignment rates are strongly correlated with prev_gpa
    data['assignment_submission_rate'] = (data['prev_gpa'] / 4.0 * 100 + np.random.normal(loc=0, scale=10, size=num_records)).clip(0, 100)
    data['midterm_score'] = (data['prev_gpa'] / 4.0 * 100 + np.random.normal(loc=0, scale=8, size=num_records)).clip(0, 100)
    
    df = pd.DataFrame(data)
    
    # Target Generation based on midterm, assignment rate, and GPA
    score = (df['prev_gpa'] * 25) * 0.3 + df['midterm_score'] * 0.4 + df['assignment_submission_rate'] * 0.3
    
    conditions = [
        (score >= 70),
        (score >= 50) & (score < 70),
        (score < 50)
    ]
    choices = ['Pass', 'At-Risk', 'Fail']
    
    df['target'] = np.select(conditions, choices, default='Fail')
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    output_path = os.path.join(os.path.dirname(__file__), 'student_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} records at {output_path}")
    print("\nClass Distribution:")
    print(df['target'].value_counts())
    print("\nFeature preview:")
    print(df.head())
