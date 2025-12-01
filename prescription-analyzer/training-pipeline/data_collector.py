import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from pathlib import Path

class DataCollector:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI'))
        self.db = self.mongo_client['prescription_analyzer']
        self.data_path = Path('ml-service/data')
    
    async def collect_feedback(self):
        """Collect feedback from database"""
        # Get feedback from last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        
        feedback_cursor = self.db.feedback.find({
            'timestamp': {'$gte': yesterday}
        })
        
        feedback_list = list(feedback_cursor)
        
        if not feedback_list:
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(feedback_list)
        
        # Save to CSV
        csv_path = self.data_path / 'feedback' / 'feedback_log.csv'
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        
        # Also save individual feedback files for detailed analysis
        for feedback in feedback_list:
            self._save_feedback_detail(feedback)
        
        return feedback_list
    
    def _save_feedback_detail(self, feedback):
        """Save detailed feedback for training"""
        feedback_id = feedback['_id']
        detail_path = self.data_path / 'feedback' / 'details' / f'{feedback_id}.json'
        
        import json
        with open(detail_path, 'w') as f:
            json.dump(feedback, f, indent=2, default=str)
    
    async def prepare_training_dataset(self):
        """Prepare dataset from collected feedback"""
        csv_path = self.data_path / 'feedback' / 'feedback_log.csv'
        df = pd.read_csv(csv_path)
        
        # Filter only corrections (user-verified data)
        training_df = df[df['feedback_type'] == 'correction']
        
        # Save to training directory
        training_path = self.data_path / 'training' / f'training_{datetime.now().strftime("%Y%m%d")}.csv'
        training_df.to_csv(training_path, index=False)
        
        return training_path