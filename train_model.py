"""
Standalone model training script for Insurance Premium Predictor
Run this separately: python train_model.py
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Train insurance premium prediction model"""
    try:
        # Import after loading environment - FIXED imports
        from src.models.trainer import ModelTrainer
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner
        from src.data.preprocessor import DataPreprocessor
        from config.config import Config
        
        print("🏥 Insurance Premium Model Training")
        print("=" * 50)
        
        print("📂 Loading data...")
        data_loader = DataLoader()
        data = data_loader.load_data()
        print(f"  ✓ Loaded {len(data):,} records")
        
        print("🧹 Cleaning data...")
        data_cleaner = DataCleaner(data)
        cleaned_data = data_cleaner.clean_data()
        print(f"  ✓ Cleaned data: {len(cleaned_data):,} records remaining")
        
        print("⚙️ Preprocessing data...")
        preprocessor = DataPreprocessor()
        preprocessor_pipeline = preprocessor.create_preprocessor()
        
        X = cleaned_data.drop(Config.TARGET, axis=1)
        y = cleaned_data[Config.TARGET]
        print(f"  ✓ Features: {X.shape[1]}, Samples: {X.shape[0]:,}")
        
        print("🤖 Training and selecting best model...")
        trainer = ModelTrainer(preprocessor_pipeline)
        trainer.train_and_select_model(X, y)
        print(f"  🏆 Best model: {trainer.best_model_name}")
        
        print("📊 Generating reports...")
        trainer.generate_reports()
        print("  ✓ Reports generated")
        
        print("💾 Saving model...")
        if trainer.save_model():
            print("  ✅ Model training completed successfully!")
            print(f"  🏆 Best model: {trainer.best_model_name}")
            print(f"  📈 R² Score: {trainer.model_metrics[trainer.best_model_name]['test_r2']:.4f}")
            return True
        else:
            print("  ❌ Failed to save model")
            return False
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training complete! You can now use the predictor.")
    else:
        print("\n💔 Training failed. Please check the logs.")
        sys.exit(1)
