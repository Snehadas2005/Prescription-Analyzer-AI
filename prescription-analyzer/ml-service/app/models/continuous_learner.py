import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
import logging

class ContinuousLearner:
    def __init__(self):
        self.model_path = Path("models/production")
        self.data_path = Path("data")
        self.version = self._load_version()
        self.logger = logging.getLogger(__name__)
        
        # Load current model
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def _load_version(self):
        """Load current model version"""
        version_file = self.model_path / "version.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return "v1.0.0"
    
    def _load_model(self):
        """Load the current production model"""
        if (self.model_path / "model.pth").exists():
            return torch.load(self.model_path / "model.pth")
        # Load base model if no production model exists
        return AutoModelForTokenClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=20  # Adjust based on your entities
        )
    
    async def trigger_training(self):
        """Trigger a new training cycle"""
        self.logger.info("Starting continuous learning cycle...")
        
        try:
            # 1. Load feedback data
            feedback_df = self._load_feedback_data()
            
            # 2. Prepare training dataset
            train_dataset = self._prepare_dataset(feedback_df)
            
            # 3. Train model
            metrics = await self._train_model(train_dataset)
            
            # 4. Evaluate performance
            if self._should_deploy(metrics):
                self._deploy_model()
                self.logger.info("New model deployed successfully!")
            else:
                self.logger.info("New model did not improve. Keeping current version.")
            
            # 5. Log metrics
            self._log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _load_feedback_data(self):
        """Load feedback data from CSV and database"""
        # Load from CSV
        csv_path = self.data_path / "feedback" / "feedback_log.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()
        
        # Filter recent feedback (last 30 days)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff_date = datetime.now() - pd.Timedelta(days=30)
        df = df[df['timestamp'] >= cutoff_date]
        
        return df
    
    def _prepare_dataset(self, feedback_df):
        """Prepare dataset from feedback"""
        # Convert feedback to training format
        # Format: {"text": "...", "labels": [...]}
        
        training_data = []
        for _, row in feedback_df.iterrows():
            if row['feedback_type'] == 'correction':
                training_data.append({
                    'text': row['extracted_text'],
                    'labels': self._parse_corrections(row['corrections'])
                })
        
        return training_data
    
    async def _train_model(self, train_dataset):
        """Train the model with new data"""
        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(3):  # Few epochs for fine-tuning
            for batch in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'timestamp': datetime.now().isoformat(),
            'samples_trained': len(train_dataset)
        }
        
        return metrics
    
    def _should_deploy(self, metrics):
        """Decide if new model should be deployed"""
        # Load previous metrics
        metrics_file = self.model_path / "metrics.json"
        if not metrics_file.exists():
            return True  # Deploy if first model
        
        import json
        with open(metrics_file) as f:
            prev_metrics = json.load(f)
        
        # Deploy if loss improved by at least 5%
        improvement = (prev_metrics['loss'] - metrics['loss']) / prev_metrics['loss']
        return improvement > 0.05
    
    def _deploy_model(self):
        """Deploy the newly trained model"""
        # Save new model
        torch.save(self.model, self.model_path / "model.pth")
        
        # Update version
        version_parts = self.version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = '.'.join(version_parts)
        
        (self.model_path / "version.txt").write_text(new_version)
        self.version = new_version
    
    def _log_metrics(self, metrics):
        """Log training metrics"""
        import json
        
        # Save metrics
        metrics_file = self.model_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Append to history
        history_file = self.data_path / "training_history.csv"
        df = pd.DataFrame([metrics])
        df.to_csv(history_file, mode='a', header=not history_file.exists(), index=False)
    
    def get_version(self):
        return self.version