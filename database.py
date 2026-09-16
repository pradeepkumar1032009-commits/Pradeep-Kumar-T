from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class SleepData(db.Model):
    __tablename__ = 'sleep_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Sensor Data
    heart_rate = db.Column(db.Float)
    spo2 = db.Column(db.Float)
    movement = db.Column(db.String(10))
    body_temp = db.Column(db.Float)
    room_temp = db.Column(db.Float)
    humidity = db.Column(db.Float)
    respiration = db.Column(db.Float)
    sleep_stage = db.Column(db.String(20))
    noise_level = db.Column(db.Float)
    
    # Derived Metrics
    heart_rate_variability = db.Column(db.Float)
    movement_frequency = db.Column(db.Float)
    deep_sleep_duration = db.Column(db.Float)
    rem_sleep_duration = db.Column(db.Float)
    
    # Predictions
    predicted_mood = db.Column(db.String(20))
    predicted_heart_risk = db.Column(db.String(20))
    mood_confidence = db.Column(db.Float)
    heart_risk_confidence = db.Column(db.Float)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'heart_rate': self.heart_rate,
            'spo2': self.spo2,
            'movement': self.movement,
            'body_temp': self.body_temp,
            'room_temp': self.room_temp,
            'respiration': self.respiration,
            'sleep_stage': self.sleep_stage,
            'noise_level': self.noise_level,
            'heart_rate_variability': self.heart_rate_variability,
            'movement_frequency': self.movement_frequency,
            'deep_sleep_duration': self.deep_sleep_duration,
            'rem_sleep_duration': self.rem_sleep_duration,
            'predicted_mood': self.predicted_mood,
            'predicted_heart_risk': self.predicted_heart_risk,
            'mood_confidence': self.mood_confidence,
            'heart_risk_confidence': self.heart_risk_confidence
        }

class DailySummary(db.Model):
    __tablename__ = 'daily_summary'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False)
    
    # Sleep Quality Metrics
    total_sleep_hours = db.Column(db.Float)
    sleep_efficiency = db.Column(db.Float)
    deep_sleep_percentage = db.Column(db.Float)
    rem_sleep_percentage = db.Column(db.Float)
    awake_count = db.Column(db.Integer)
    
    # Health Metrics
    avg_heart_rate = db.Column(db.Float)
    avg_spo2 = db.Column(db.Float)
    min_spo2 = db.Column(db.Float)
    heart_rate_variability = db.Column(db.Float)
    avg_respiration = db.Column(db.Float)
    
    # Risk Scores
    mood_score = db.Column(db.String(20))
    heart_risk_score = db.Column(db.String(20))
    overall_health_score = db.Column(db.Integer)
    
    # Recommendations
    recommendations = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'date': self.date.isoformat() if self.date else None,
            'total_sleep_hours': self.total_sleep_hours,
            'sleep_efficiency': self.sleep_efficiency,
            'deep_sleep_percentage': self.deep_sleep_percentage,
            'rem_sleep_percentage': self.rem_sleep_percentage,
            'awake_count': self.awake_count,
            'avg_heart_rate': self.avg_heart_rate,
            'avg_spo2': self.avg_spo2,
            'min_spo2': self.min_spo2,
            'heart_rate_variability': self.heart_rate_variability,
            'avg_respiration': self.avg_respiration,
            'mood_score': self.mood_score,
            'heart_risk_score': self.heart_risk_score,
            'overall_health_score': self.overall_health_score,
            'recommendations': self.recommendations
        }
