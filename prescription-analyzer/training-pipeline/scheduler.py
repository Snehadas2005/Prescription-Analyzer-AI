import schedule
import time
import asyncio
from training_pipeline.data_collector import DataCollector
from training_pipeline.model_updater import ModelUpdater
from training_pipeline.performance_monitor import PerformanceMonitor

class TrainingScheduler:
    def __init__(self):
        self.data_collector = DataCollector()
        self.model_updater = ModelUpdater()
        self.performance_monitor = PerformanceMonitor()
    
    async def daily_training_job(self):
        """Run daily training job"""
        print(f"Starting daily training job at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. Collect new feedback data
            new_samples = await self.data_collector.collect_feedback()
            print(f"Collected {len(new_samples)} new samples")
            
            # 2. Check if we have enough data to train
            if len(new_samples) < 10:
                print("Not enough new samples. Skipping training.")
                return
            
            # 3. Trigger model training
            metrics = await self.model_updater.train()
            print(f"Training completed. Metrics: {metrics}")
            
            # 4. Monitor performance
            await self.performance_monitor.log_metrics(metrics)
            
            # 5. Send notification if model improved
            if metrics.get('improved', False):
                await self.performance_monitor.notify_improvement(metrics)
            
        except Exception as e:
            print(f"Training job failed: {e}")
            await self.performance_monitor.notify_error(str(e))
    
    def run(self):
        """Run the scheduler"""
        # Schedule daily training at 2 AM
        schedule.every().day.at("02:00").do(
            lambda: asyncio.run(self.daily_training_job())
        )
        
        # Run immediately on startup for testing
        asyncio.run(self.daily_training_job())
        
        print("Training scheduler started. Press Ctrl+C to stop.")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    scheduler = TrainingScheduler()
    scheduler.run()