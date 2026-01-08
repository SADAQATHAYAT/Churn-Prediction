import pandas as pd
import numpy as np
import os

def generate_customer_churn_data(n_samples=50000):
    """
    Generates a synthetic customer churn dataset.
    """
    np.random.seed(42)
    
    # Customer Demographics
    customer_ids = [f'CUST-{i:05d}' for i in range(1, n_samples + 1)]
    gender = np.random.choice(['Male', 'Female'], n_samples)
    age = np.random.randint(18, 80, n_samples)
    senior_citizen = np.where(age > 65, 1, 0)
    partner = np.random.choice(['Yes', 'No'], n_samples)
    dependents = np.random.choice(['Yes', 'No'], n_samples)
    
    # Account Information
    tenure = np.random.randint(1, 73, n_samples)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples)
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
        n_samples
    )
    
    # Services Info
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
    multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
    
    # Usage Metrics (Numerical)
    monthly_charges = np.random.uniform(18.25, 118.75, n_samples)
    total_charges = monthly_charges * tenure + np.random.uniform(0, 100, n_samples) # Add some noise
    
    # Target Variable Logic (Simulating Churn)
    # Higher churn probability for:
    # - Month-to-month contract
    # - Fiber optic internet
    # - Short tenure
    # - Electronic check payment
    
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.4, 0)
    churn_prob += np.where(contract == 'One year', 0.1, 0)
    churn_prob += np.where(contract == 'Two year', 0.05, 0)
    
    churn_prob += np.where(internet_service == 'Fiber optic', 0.2, 0)
    churn_prob += np.where(tenure < 12, 0.2, 0)
    churn_prob += np.where(payment_method == 'Electronic check', 0.1, 0)
    
    # Normalize probabilities
    churn_prob = np.clip(churn_prob, 0, 0.9)
    
    # Generate labels
    churn = np.random.binomial(1, churn_prob)
    churn_labels = np.where(churn == 1, 'Yes', 'No')
    
    # Create DataFrame
    data = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn_labels
    })
    
    # Introduce some missing values for realism (TotalCharges is a common culprit)
    mask = np.random.random(n_samples) < 0.005 # 0.5% missing
    data.loc[mask, 'TotalCharges'] = np.nan
    
    return data

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    
    print("Generating synthetic data...")
    df = generate_customer_churn_data(50000)
    output_path = 'data/customer_churn.csv'
    df.to_csv(output_path, index=False)
    print(f"Data generated and saved to {output_path}")
    print(df.head())
