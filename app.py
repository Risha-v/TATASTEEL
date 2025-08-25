import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.models.predictor import InsurancePredictor
from src.models.trainer import ModelTrainer
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.preprocessor import DataPreprocessor
from config.config import Config
from config.paths import Paths
from dotenv import load_dotenv
import os
import time

# Page configuration
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    .health-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def load_environment():
    """Load environment variables"""
    load_dotenv()

def train_model_if_needed():
    """Train model if it doesn't exist"""
    if not os.path.exists(Paths.MODEL_SAVE_PATH):
        st.warning("Model not found. Training new model...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load data
            status_text.text("Loading data...")
            progress_bar.progress(20)
            data_loader = DataLoader()
            data = data_loader.load_data()
            
            # Clean data
            status_text.text("Cleaning data...")
            progress_bar.progress(40)
            data_cleaner = DataCleaner(data)
            cleaned_data = data_cleaner.clean_data()
            
            # Preprocess
            status_text.text("Preprocessing data...")
            progress_bar.progress(60)
            preprocessor = DataPreprocessor()
            preprocessor_pipeline = preprocessor.create_preprocessor()
            
            X = cleaned_data.drop(Config.TARGET, axis=1)
            y = cleaned_data[Config.TARGET]
            
            # Train model
            status_text.text("Training models...")
            progress_bar.progress(80)
            trainer = ModelTrainer(preprocessor_pipeline)
            trainer.train_and_select_model(X, y)
            
            # Generate reports
            status_text.text("Generating reports...")
            progress_bar.progress(90)
            trainer.generate_reports()
            
            # Save model
            if trainer.save_model():
                progress_bar.progress(100)
                status_text.text("Model trained successfully!")
                st.success("✅ Model training completed!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to save trained model")
                return False
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return False
        
    return True

def create_input_form():
    """Create user input form"""
    st.subheader("📋 Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 35)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    
    with col2:
        sex = st.selectbox("Gender", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    return {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }

def display_prediction_results(prediction, profile):
    """Display prediction results in a beautiful format"""
    # Main prediction
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin: 0; text-align: center;">Annual Premium Prediction</h2>
        <h1 style="margin: 0.5rem 0; text-align: center; font-size: 3rem;">${prediction:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Financial breakdown
    financial = profile['financial']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly", financial['monthly'], help="Monthly premium payment")
    with col2:
        st.metric("Weekly", financial['weekly'], help="Weekly premium cost")
    with col3:
        st.metric("Daily", financial['daily'], help="Daily premium cost")
    with col4:
        st.metric("Risk Tier", financial['risk_tier'], help="Risk classification")
    
    # Health analysis
    st.subheader("🏥 Health Analysis")
    health = profile['health']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**BMI:** {health['bmi']}")
        st.write(f"**Smoker:** {health['smoker']}")
        st.write(f"**Risk Factors:** {', '.join(health['risk_factors'])}")
    
    with col2:
        score = health['overall_health_score']
        color = "#ff4444" if score < 5 else "#ffaa00" if score < 7 else "#44ff44"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: {color}; border-radius: 10px; color: white;">
            <div class="health-score">{score:.1f}/10</div>
            <div>Health Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    recommendations = profile['recommendations']
    if recommendations:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['recommendation'].title()}", expanded=i==1):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Category:** {rec['category']}")
                    st.write(f"**Impact:** {rec['impact']}")
                    st.write(f"**Details:** {rec['details']}")
                with col2:
                    st.metric("Potential Savings", rec['potential_savings'])
                    st.write(f"**Timeframe:** {rec['timeframe']}")

def main():
    """Main Streamlit application"""
    # Load environment
    load_environment()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Insurance Premium Predictor</h1>', unsafe_allow_html=True)
    
    # Train model if needed
    if not train_model_if_needed():
        st.error("Failed to initialize model. Please check your data and configuration.")
        return
    
    # Load predictor
    predictor = InsurancePredictor()
    if not predictor.load_model():
        st.error("Failed to load model. Please train the model first.")
        return
    
    # Display model info
    st.info(f"🤖 Using {predictor.model_name} model for predictions")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        # Input form
        input_data = create_input_form()
        
        # Prediction button
        if st.button("🔮 Predict Premium", type="primary"):
            try:
                with st.spinner("Calculating prediction..."):
                    prediction, profile = predictor.predict(input_data)
                
                # Display results
                display_prediction_results(prediction, profile)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    with tab2:
        st.subheader("📈 Model Performance")
        
        # Try to load and display model metrics
        try:
            metrics_path = os.path.join(Paths.REPORTS_DIR, 'model_performance.csv')
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path, index_col=0)
                
                # Display metrics table
                st.dataframe(metrics_df.round(4))
                
                # Create performance chart
                fig = px.bar(
                    x=metrics_df.index,
                    y=metrics_df['test_r2'],
                    title="Model Performance Comparison (R² Score)",
                    labels={'x': 'Model', 'y': 'R² Score'}
                )
                st.plotly_chart(fig)
            else:
                st.warning("Performance metrics not available. Train the model to generate reports.")
        except Exception as e:
            st.error(f"Could not load analytics: {str(e)}")
    
    with tab3:
        st.subheader("ℹ️ About This Application")
        st.markdown("""
        This Insurance Premium Prediction System uses advanced machine learning to predict health insurance premiums based on:
        
        - **Demographics**: Age, gender, region
        - **Health Factors**: BMI, smoking status
        - **Family Information**: Number of children
        
        **Key Features:**
        - 🤖 Automated model selection from 5 ML algorithms
        - 📊 Comprehensive risk analysis
        - 💡 Personalized recommendations for premium reduction
        - 📈 Real-time predictions with detailed breakdowns
        
        **Models Used:**
        - Random Forest
        - XGBoost  
        - LightGBM
        - CatBoost
        - Decision Tree
        
        The system automatically selects the best performing model based on cross-validation results.
        """)

if __name__ == "__main__":
    main()
