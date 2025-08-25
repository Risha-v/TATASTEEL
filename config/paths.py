import os
from dotenv import load_dotenv

load_dotenv()

class Paths:
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data paths
    RAW_DATA = os.getenv('RAW_DATA_PATH', os.path.join(BASE_DIR, 'data/raw/insurance.csv'))
    PROCESSED_DATA = os.getenv('PROCESSED_DATA_PATH', os.path.join(BASE_DIR, 'data/processed/processed_data.csv'))
    
    # Model paths
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', os.path.join(BASE_DIR, 'models/insurance_model.pkl'))
    
    # Log and report paths
    LOG_FILE = os.getenv('LOG_FILE', os.path.join(BASE_DIR, 'logs/app.log'))
    REPORTS_DIR = os.getenv('REPORTS_DIR', os.path.join(BASE_DIR, 'reports/'))
