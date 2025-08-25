import os
import pandas as pd
from config.paths import Paths
import joblib
from src.utils.logger import get_logger
from config.config import Config
import numpy as np

logger = get_logger(__name__)

class InsurancePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_name = ""
        self.config = None

    def load_model(self):
        """Load a saved model with error handling"""
        try:
            if not Paths.MODEL_SAVE_PATH or not os.path.exists(Paths.MODEL_SAVE_PATH):
                logger.error(f"Model file not found at {Paths.MODEL_SAVE_PATH}")
                return False

            # Load model data
            model_data = joblib.load(Paths.MODEL_SAVE_PATH)
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', None)
            self.model_name = model_data.get('model_name', "Unknown Model")
            self.config = model_data.get('config', {})

            logger.info(f"{self.model_name} model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict(self, input_data):
        """Make prediction with comprehensive customer profiling"""
        try:
            # Validate input
            validation_errors = self._validate_input(input_data)
            if validation_errors:
                raise ValueError(f"Input validation failed: {validation_errors}")

            # Convert to DataFrame to maintain feature names for LightGBM/CatBoost
            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = self.model.predict(input_df)[0]

            # Ensure positive prediction
            prediction = max(0, prediction)

            logger.info(f"Prediction made: ${prediction:.2f} for input: {input_data}")

            # Generate comprehensive profile
            profile = self._generate_profile(input_data, prediction)

            return prediction, profile

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _validate_input(self, input_data):
        """Validate input data against configuration rules"""
        errors = []

        # Check required fields
        required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")

        if errors:  # Return early if fields are missing
            return errors

        # Validate numerical ranges
        for field in ['age', 'bmi', 'children']:
            value = input_data.get(field)
            if value is not None:
                min_val, max_val = Config.VALIDATION_RULES[field]
                if not (min_val <= value <= max_val):
                    errors.append(f"{field} must be between {min_val} and {max_val}")

        # Validate categorical values
        for field in ['sex', 'smoker', 'region']:
            value = input_data.get(field)
            if value is not None:
                allowed_values = Config.VALIDATION_RULES[field]
                if value.lower() not in [v.lower() for v in allowed_values]:
                    errors.append(f"{field} must be one of: {', '.join(allowed_values)}")

        return errors

    def _generate_profile(self, input_data, prediction):
        """Generate comprehensive customer profile with insights"""
        return {
            'summary': self._get_summary(input_data),
            'health': self._get_health_analysis(input_data),
            'financial': self._get_financial_breakdown(prediction),
            'recommendations': self._get_recommendations(input_data, prediction)
        }

    def _get_summary(self, data):
        """Generate demographic summary"""
        age = data['age']
        age_group = "Young Adult" if age < 30 else "Middle-aged" if age < 50 else "Senior"

        family_status = "No children"
        if data['children'] == 1:
            family_status = "1 child"
        elif data['children'] > 1:
            family_status = f"{data['children']} children"

        return {
            'age': f"{age} years ({age_group})",
            'gender': data['sex'].title(),
            'region': data['region'].replace('_', ' ').title(),
            'children': data['children'],
            'family_status': family_status
        }

    def _get_health_analysis(self, data):
        """Analyze health factors and risk assessment"""
        bmi = data['bmi']

        # BMI categorization
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_risk = "Medium"
        elif bmi < 25:
            bmi_status = "Normal Weight"
            bmi_risk = "Low"
        elif bmi < 30:
            bmi_status = "Overweight"
            bmi_risk = "Medium"
        else:
            bmi_status = "Obese"
            bmi_risk = "High"

        # Identify risk factors
        risk_factors = []
        if data['smoker'] == 'yes':
            risk_factors.append("Smoking (Major Risk)")

        if bmi >= 30:
            risk_factors.append("Obesity")
        elif bmi >= 25:
            risk_factors.append("Overweight")

        if data['age'] >= 50:
            risk_factors.append("Age-related Risk")

        return {
            'bmi': f"{bmi:.1f} ({bmi_status})",
            'bmi_risk': bmi_risk,
            'smoker': "Yes" if data['smoker'] == 'yes' else "No",
            'smoking_risk': "Very High" if data['smoker'] == 'yes' else "None",
            'risk_factors': risk_factors if risk_factors else ["Low Risk Profile"],
            'overall_health_score': self._calculate_health_score(data)
        }

    def _calculate_health_score(self, data):
        """Calculate a health score from 1-10 (10 being healthiest)"""
        score = 10

        # Smoking penalty
        if data['smoker'] == 'yes':
            score -= 4

        # BMI penalty
        bmi = data['bmi']
        if bmi < 18.5 or bmi >= 30:
            score -= 2
        elif bmi >= 25:
            score -= 1

        # Age penalty
        if data['age'] >= 60:
            score -= 1
        elif data['age'] >= 50:
            score -= 0.5

        return max(1, min(10, score))

    def _get_financial_breakdown(self, prediction):
        """Generate detailed financial breakdown"""
        annual = prediction
        monthly = annual / 12
        weekly = annual / 52
        daily = annual / 365

        # Risk tier classification
        if annual < 5000:
            risk_tier = "Low Risk"
            tier_desc = "Excellent health profile"
        elif annual < 15000:
            risk_tier = "Medium Risk"
            tier_desc = "Average health profile"
        elif annual < 30000:
            risk_tier = "High Risk"
            tier_desc = "Elevated health risks"
        else:
            risk_tier = "Very High Risk"
            tier_desc = "Significant health concerns"

        return {
            'annual': f"${annual:,.2f}",
            'monthly': f"${monthly:,.2f}",
            'weekly': f"${weekly:,.2f}",
            'daily': f"${daily:.2f}",
            'risk_tier': risk_tier,
            'tier_description': tier_desc,
            'premium_category': self._get_premium_category(annual)
        }

    def _get_premium_category(self, annual_premium):
        """Categorize premium level"""
        if annual_premium < 3000:
            return "Budget-Friendly"
        elif annual_premium < 8000:
            return "Standard"
        elif annual_premium < 15000:
            return "Premium"
        else:
            return "High-Cost"

    def _get_recommendations(self, data, prediction):
        """Generate personalized recommendations for premium reduction"""
        recommendations = []

        # Smoking recommendations
        if data['smoker'] == 'yes':
            potential_savings = prediction * 0.4
            recommendations.append({
                'category': 'Lifestyle',
                'recommendation': 'Quit smoking',
                'impact': 'High',
                'potential_savings': f"${potential_savings:,.2f}",
                'timeframe': '12 months',
                'details': 'Smoking is the largest risk factor. Quitting can reduce premiums by up to 50%.'
            })

        # BMI recommendations
        if data['bmi'] >= 30:
            potential_savings = prediction * 0.15
            recommendations.append({
                'category': 'Health',
                'recommendation': 'Weight management program',
                'impact': 'Medium',
                'potential_savings': f"${potential_savings:,.2f}",
                'timeframe': '6-12 months',
                'details': 'Reducing BMI below 25 can significantly lower premiums.'
            })
        elif data['bmi'] >= 25:
            potential_savings = prediction * 0.08
            recommendations.append({
                'category': 'Health',
                'recommendation': 'Maintain healthy weight',
                'impact': 'Low-Medium',
                'potential_savings': f"${potential_savings:,.2f}",
                'timeframe': '3-6 months',
                'details': 'Small weight reduction can help optimize your premium rates.'
            })

        # If no major recommendations
        if not recommendations:
            recommendations.append({
                'category': 'Status',
                'recommendation': 'Maintain current lifestyle',
                'impact': 'Positive',
                'potential_savings': '$0.00',
                'timeframe': 'Ongoing',
                'details': 'You qualify for our best rates! Keep up the healthy lifestyle.'
            })

        return recommendations
