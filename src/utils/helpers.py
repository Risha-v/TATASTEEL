from config.config import Config
import re

class InputValidator:
    @staticmethod
    def validate_input(input_data):
        """Validate user input against configuration rules"""
        errors = []
        
        # Check for required fields
        required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return errors
        
        # Validate numerical fields
        for field in ['age', 'bmi', 'children']:
            value = input_data.get(field)
            if value is not None:
                try:
                    # Convert to appropriate type
                    if field in ['age', 'children']:
                        value = int(value)
                    else:
                        value = float(value)
                    
                    # Check range
                    min_val, max_val = Config.VALIDATION_RULES[field]
                    if not (min_val <= value <= max_val):
                        errors.append(f"{field} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Validate categorical fields
        for field in ['sex', 'smoker', 'region']:
            value = input_data.get(field)
            if value is not None:
                allowed_values = Config.VALIDATION_RULES[field]
                if value.lower() not in [v.lower() for v in allowed_values]:
                    errors.append(f"{field} must be one of: {', '.join(allowed_values)}")
        
        return errors

    @staticmethod
    def clean_input(input_str):
        """Clean and normalize input string"""
        return input_str.strip().lower()

    @staticmethod
    def parse_numeric_input(input_str, input_type=float):
        """Parse numeric input with error handling"""
        try:
            return input_type(input_str.strip())
        except ValueError:
            raise ValueError(f"Invalid {input_type.__name__} value: {input_str}")

class DataFormatter:
    @staticmethod
    def format_currency(amount):
        """Format currency with proper formatting"""
        return f"${amount:,.2f}"

    @staticmethod
    def format_percentage(value):
        """Format percentage with proper formatting"""
        return f"{value:.1f}%"

    @staticmethod
    def format_bmi(bmi):
        """Format BMI with categorization"""
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        return f"{bmi:.1f} ({category})"

class HealthAnalyzer:
    @staticmethod
    def get_bmi_category(bmi):
        """Get BMI category and risk level"""
        if bmi < 18.5:
            return "Underweight", "Medium"
        elif bmi < 25:
            return "Normal Weight", "Low"
        elif bmi < 30:
            return "Overweight", "Medium"
        else:
            return "Obese", "High"

    @staticmethod
    def calculate_smoking_impact(base_premium):
        """Calculate estimated smoking impact on premium"""
        return base_premium * 1.5  # Smoking typically increases premiums by ~50%

    @staticmethod
    def get_age_risk_category(age):
        """Get age-based risk category"""
        if age < 30:
            return "Low Risk"
        elif age < 45:
            return "Medium Risk"
        elif age < 60:
            return "High Risk"
        else:
            return "Very High Risk"
