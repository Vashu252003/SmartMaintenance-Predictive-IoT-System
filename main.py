"""
Predictive Maintenance System for IoT Machines
Main application entry point with data generation, model training, and API serving
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path

# Data Generation Module
class SensorDataGenerator:
    """Generate synthetic sensor data for IoT machines"""
    
    def __init__(self, num_machines=5, days=365):
        self.num_machines = num_machines
        self.days = days
        self.data = None
        
    def generate_data(self):
        """Generate synthetic sensor readings with failure patterns"""
        np.random.seed(42)
        
        records = []
        start_date = datetime.now() - timedelta(days=self.days)
        
        for machine_id in range(1, self.num_machines + 1):
            # Simulate failure at random intervals
            failure_dates = sorted(np.random.choice(
                range(self.days), 
                size=np.random.randint(2, 5), 
                replace=False
            ))
            
            for day in range(self.days):
                current_date = start_date + timedelta(days=day)
                
                # Check proximity to failure
                days_to_failure = min([abs(day - f) for f in failure_dates])
                is_failing = days_to_failure < 7
                
                # Generate readings (degraded when approaching failure)
                if is_failing:
                    temperature = np.random.normal(85 + (7 - days_to_failure) * 3, 5)
                    vibration = np.random.normal(0.8 + (7 - days_to_failure) * 0.05, 0.1)
                    voltage = np.random.normal(220 - (7 - days_to_failure) * 2, 3)
                    pressure = np.random.normal(95 - (7 - days_to_failure) * 2, 5)
                    rpm = np.random.normal(1450 - (7 - days_to_failure) * 10, 20)
                else:
                    temperature = np.random.normal(75, 3)
                    vibration = np.random.normal(0.5, 0.05)
                    voltage = np.random.normal(220, 2)
                    pressure = np.random.normal(100, 3)
                    rpm = np.random.normal(1500, 15)
                
                records.append({
                    'timestamp': current_date,
                    'machine_id': f'MACHINE_{machine_id:03d}',
                    'temperature': max(0, temperature),
                    'vibration': max(0, vibration),
                    'voltage': max(0, voltage),
                    'pressure': max(0, pressure),
                    'rpm': max(0, rpm),
                    'failure': 1 if day in failure_dates else 0
                })
        
        self.data = pd.DataFrame(records)
        return self.data
    
    def save_data(self, filepath='data/sensor_data.csv'):
        """Save generated data to CSV"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        

# Feature Engineering Module
class FeatureEngineer:
    """Create features for predictive modeling"""
    
    def __init__(self, data):
        self.data = data.copy()
        
    def create_features(self):
        """Engineer features from raw sensor data"""
        df = self.data.copy()
        
        # Sort by machine and timestamp
        df = df.sort_values(['machine_id', 'timestamp'])
        
        # Rolling statistics (7-day window)
        for col in ['temperature', 'vibration', 'voltage', 'pressure', 'rpm']:
            df[f'{col}_roll_mean'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f'{col}_roll_std'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).std()
            )
        
        # Lag features (previous day values)
        for col in ['temperature', 'vibration', 'voltage', 'pressure', 'rpm']:
            df[f'{col}_lag1'] = df.groupby('machine_id')[col].shift(1)
            df[f'{col}_lag3'] = df.groupby('machine_id')[col].shift(3)
        
        # Rate of change
        for col in ['temperature', 'vibration', 'voltage']:
            df[f'{col}_rate'] = df.groupby('machine_id')[col].diff()
        
        # Interaction features
        df['temp_vibration'] = df['temperature'] * df['vibration']
        df['voltage_rpm'] = df['voltage'] * df['rpm']
        
        # Fill NaN values from lag/rolling features
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    

# Model Training Module
class PredictiveModel:
    """Train and evaluate predictive maintenance models"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.scaler = None
        
    def prepare_data(self, df, test_size=0.2):
        """Prepare train/test splits"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Select feature columns
        self.feature_cols = [col for col in df.columns if col not in 
                            ['timestamp', 'machine_id', 'failure']]
        
        X = df[self.feature_cols]
        y = df['failure']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train Random Forest classifier"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.class_weight import compute_class_weight
        
        # Handle class imbalance
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n=== Model Evaluation ===")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_imp.head(10))
        
        return y_pred, y_pred_proba
    
    def save_model(self, model_dir='models'):
        """Save trained model and scaler"""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        with open(f'{model_dir}/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(f'{model_dir}/feature_cols.json', 'w') as f:
            json.dump(self.feature_cols, f)
        
        print(f"Model saved to {model_dir}/")
        

# Main execution
if __name__ == "__main__":
    print("=== Predictive Maintenance System ===\n")
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating sensor data...")
    generator = SensorDataGenerator(num_machines=10, days=365)
    raw_data = generator.generate_data()
    generator.save_data()
    print(f"Generated {len(raw_data)} records for {raw_data['machine_id'].nunique()} machines")
    print(f"Failure rate: {raw_data['failure'].mean()*100:.2f}%\n")
    
    # Step 2: Feature engineering
    print("Step 2: Engineering features...")
    engineer = FeatureEngineer(raw_data)
    processed_data = engineer.create_features()
    processed_data.to_csv('data/processed_data.csv', index=False)
    print(f"Created {len(processed_data.columns)} features\n")
    
    # Step 3: Train model
    print("Step 3: Training predictive model...")
    model = PredictiveModel()
    X_train, X_test, y_train, y_test = model.prepare_data(processed_data)
    model.train_model(X_train, y_train)
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluating model performance...")
    model.evaluate_model(X_test, y_test)
    
    # Step 5: Save model
    print("\nStep 5: Saving model artifacts...")
    model.save_model()
    
    print("\n=== Setup Complete! ===")
    print("Next steps:")
    print("1. Run 'python api.py' to start the prediction API")
    print("2. Run 'streamlit run dashboard.py' to view the dashboard")