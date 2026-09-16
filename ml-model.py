import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HealthPredictor:
    def __init__(self):
        self.mood_model = None
        self.heart_model = None
        self.mood_scaler = StandardScaler()
        self.heart_scaler = StandardScaler()
        self.mood_encoder = LabelEncoder()
        self.heart_encoder = LabelEncoder()
        
    def generate_training_data(self, num_samples=1000, use_csv=True):
        df = None
        if use_csv:
            raw_frames = []
            for csv_path in ['data/iot_sleep_dataset_new.csv', 'data/iot_sleep_dataset_movement.csv']:
                try:
                    raw_df = pd.read_csv(csv_path)
                    raw_frames.append(raw_df)
                    print(f"Loading data from {csv_path}...")
                except FileNotFoundError:
                    continue

            if raw_frames:
                raw_df = pd.concat(raw_frames, ignore_index=True)
                rename_map = {
                    'Heart_Rate': 'HeartRate',
                    'Temperature': 'Temp',
                    'SPO2': 'SpO2',
                    'Oxygen': 'SpO2',
                    'RespirationRate': 'Respiration'
                }
                for old_col, new_col in rename_map.items():
                    if old_col in raw_df.columns:
                        if new_col in raw_df.columns:
                            raw_df[new_col] = raw_df[new_col].fillna(raw_df[old_col])
                            raw_df = raw_df.drop(columns=[old_col])
                        else:
                            raw_df = raw_df.rename(columns={old_col: new_col})

                raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()].copy()

                if 'HeartRate' not in raw_df.columns and 'Heart_Rate' in raw_df.columns:
                    raw_df['HeartRate'] = raw_df['Heart_Rate']
                if 'Movement' not in raw_df.columns:
                    raw_df['Movement'] = 'LOW'
                if 'Temp' not in raw_df.columns:
                    raw_df['Temp'] = 31.0
                if 'Humidity' not in raw_df.columns:
                    raw_df['Humidity'] = 50.0

                df = pd.DataFrame()
                df['avg_heart_rate'] = pd.to_numeric(raw_df['HeartRate'], errors='coerce').fillna(70.0)

                movement_map = {
                    'HIGH': 8.0, 'Y': 8.0,
                    'MID': 4.0, 'M': 4.0,
                    'LOW': 1.0, 'N': 1.0, 'Z': 1.0
                }
                movement_clean = raw_df['Movement'].astype(str).str.upper()
                df['movement_frequency'] = movement_clean.map(movement_map).fillna(1.0)
                df['movement_frequency'] += np.random.normal(0, 0.25, len(df))
                df['movement_frequency'] = df['movement_frequency'].clip(0.5, 10.0)

                df['awake_count'] = (df['movement_frequency'] / 1.8).astype(int).clip(0, 10)
                temp = pd.to_numeric(raw_df['Temp'], errors='coerce').fillna(31.0)
                humidity = pd.to_numeric(raw_df['Humidity'], errors='coerce').fillna(50.0)

                if 'SpO2' in raw_df.columns:
                    df['spo2_avg'] = pd.to_numeric(raw_df['SpO2'], errors='coerce').fillna(97.0)
                else:
                    df['spo2_avg'] = 99.0 - (df['avg_heart_rate'] - 70.0) * 0.08 - (df['movement_frequency'] - 1.0) * 0.4
                df['spo2_avg'] = df['spo2_avg'].clip(85.0, 99.5)

                if 'Respiration' in raw_df.columns:
                    df['respiration_avg'] = pd.to_numeric(raw_df['Respiration'], errors='coerce').fillna(16.0)
                else:
                    df['respiration_avg'] = 12.0 + (df['avg_heart_rate'] - 60.0) * 0.08 + (df['movement_frequency'] - 1.0) * 0.25
                df['respiration_avg'] = df['respiration_avg'].clip(10.0, 28.0)

                df['heart_rate_variability'] = (
                    75.0 - (df['avg_heart_rate'] - 70.0) * 0.5 - (df['movement_frequency'] - 1.0) * 2.0
                    + np.random.normal(0, 3.0, len(df))
                ).clip(8.0, 100.0)

                df['temperature_variance'] = (
                    np.abs(temp - 31.0) * 0.15 + (df['movement_frequency'] / 12.0) + np.random.uniform(0.05, 0.25, len(df))
                ).clip(0.05, 3.0)

                df['deep_sleep_percentage'] = (
                    35.0 - df['movement_frequency'] * 2.4 - df['awake_count'] * 1.2 + np.random.normal(0, 2.0, len(df))
                ).clip(3.0, 55.0)
                df['rem_sleep_percentage'] = (
                    25.0 - df['awake_count'] * 1.1 + np.random.normal(0, 2.0, len(df))
                ).clip(5.0, 35.0)
                df['humidity_avg'] = humidity.clip(25.0, 80.0)
            else:
                print("CSV not found, using synthetic data")
                df = None

        if df is None:
            np.random.seed(42)
            data = {
                'avg_heart_rate': np.random.normal(70, 10, num_samples),
                'heart_rate_variability': np.random.normal(50, 15, num_samples),
                'spo2_avg': np.random.normal(97, 1.5, num_samples),
                'movement_frequency': np.random.exponential(2, num_samples),
                'deep_sleep_percentage': np.random.uniform(10, 40, num_samples),
                'rem_sleep_percentage': np.random.uniform(15, 30, num_samples),
                'awake_count': np.random.poisson(3, num_samples),
                'respiration_avg': np.random.normal(16, 3, num_samples),
                'temperature_variance': np.random.uniform(0.1, 1.5, num_samples)
            }
            df = pd.DataFrame(data)
        
        mood_conditions = [
            (df['deep_sleep_percentage'] > 30) & (df['awake_count'] < 3),
            (df['heart_rate_variability'] < 40) | (df['awake_count'] > 5),
            (df['movement_frequency'] > 5) | ((df['avg_heart_rate'] > 105) & (df['deep_sleep_percentage'] < 18))
        ]
        mood_choices = ['Happy', 'Stressed', 'Anxious']
        df['mood_label'] = np.select(mood_conditions, mood_choices, default='Neutral')
        
        heart_conditions = [
            (df['avg_heart_rate'] >= 120)

            | ((df['avg_heart_rate'] >= 100) & ((df['movement_frequency'] >= 6.0) | (df['spo2_avg'] < 94.0)))
            | (df['spo2_avg'] < 91.0)
            | (df['respiration_avg'] > 22.0),
            (df['avg_heart_rate'].between(90, 119))
            | (df['movement_frequency'].between(3.5, 6.0))
            | (df['spo2_avg'].between(94.0, 96.0, inclusive='left'))
            | (df['heart_rate_variability'] < 35.0),
            (df['avg_heart_rate'] < 90)
            & (df['movement_frequency'] < 3.5)
            & (df['spo2_avg'] >= 96.0)
            & (df['heart_rate_variability'] >= 35.0)
        ]
        heart_choices = ['High', 'Medium', 'Low']
        df['heart_risk'] = np.select(heart_conditions, heart_choices, default='Medium')
        
        return df
    
    def train_mood_model(self, df):
        mood_features = ['avg_heart_rate', 'heart_rate_variability', 'movement_frequency', 'deep_sleep_percentage', 'rem_sleep_percentage', 'awake_count']
        X = df[mood_features]
        y = self.mood_encoder.fit_transform(df['mood_label'])
        X_scaled = self.mood_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        self.mood_model = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42)
        self.mood_model.fit(X_train, y_train)
        return accuracy_score(y_test, self.mood_model.predict(X_test))
    
    def train_heart_model(self, df):
        heart_features = ['avg_heart_rate', 'heart_rate_variability', 'spo2_avg', 'respiration_avg', 'temperature_variance']
        X = df[heart_features]
        y = self.heart_encoder.fit_transform(df['heart_risk'])
        X_scaled = self.heart_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        self.heart_model = RandomForestClassifier(n_estimators=220, max_depth=14, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42)
        self.heart_model.fit(X_train, y_train)
        return accuracy_score(y_test, self.heart_model.predict(X_test))
    
    def train_models(self):
        print("🔄 Generating training data...")
        df = self.generate_training_data(1000)
        mood_acc = self.train_mood_model(df)
        heart_acc = self.train_heart_model(df)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.mood_model, 'models/mood_model.pkl')
        joblib.dump(self.heart_model, 'models/heart_model.pkl')
        joblib.dump(self.mood_scaler, 'models/mood_scaler.pkl')
        joblib.dump(self.heart_scaler, 'models/heart_scaler.pkl')
        joblib.dump(self.mood_encoder, 'models/mood_encoder.pkl')
        joblib.dump(self.heart_encoder, 'models/heart_encoder.pkl')
        return mood_acc, heart_acc
    
    def predict(self, features):
        if self.mood_model is None:
            self.load_models()
        mood_features = np.array([[features['avg_heart_rate'], features['heart_rate_variability'], features['movement_frequency'], features['deep_sleep_percentage'], features['rem_sleep_percentage'], features['awake_count']]])
        heart_features = np.array([[features['avg_heart_rate'], features['heart_rate_variability'], features['spo2_avg'], features['respiration_avg'], features['temperature_variance']]])
        mood_label = self.mood_encoder.inverse_transform(self.mood_model.predict(self.mood_scaler.transform(mood_features)))
        heart_label = self.heart_encoder.inverse_transform(self.heart_model.predict(self.heart_scaler.transform(heart_features)))
        mood_proba = self.mood_model.predict_proba(self.mood_scaler.transform(mood_features))[0]heart_proba = self.heart_model.predict_proba(self.heart_scaler.transform(heart_features))[0]
        return {'mood': mood_label[0], 'heart_risk': heart_label[0],'mood_probabilities': dict(zip(self.mood_encoder.classes_, mood_proba)),'heart_probabilities': dict(zip(self.heart_encoder.classes_, heart_proba))}
    def load_models(self):
        try:
          self.mood_model = joblib.load('models/mood_model.pkl')
          self.heart_model = joblib.load('models/heart_model.pkl')
          self.mood_scaler = joblib.load('models/mood_scaler.pkl')
          self.heart_scaler = joblib.load('models/heart_scaler.pkl')
          self.mood_encoder = joblib.load('models/mood_encoder.pkl')
          self.heart_encoder = joblib.load('models/heart_encoder.pkl')
        except:
          self.train_models()
if name == "main":
  predictor = HealthPredictor()
predictor.train_models()
