import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings

from config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InputValidator:
    """
    Comprehensive input validation for insurance premium prediction
    """
    
    @staticmethod
    def validate_prediction_input(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation for prediction input data
        
        Args:
            input_data: Dictionary containing input features
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check for required fields
            errors.extend(InputValidator._validate_required_fields(input_data))
            
            if errors:  # If required fields are missing, return early
                return False, errors
            
            # Validate data types and ranges
            errors.extend(InputValidator._validate_numerical_fields(input_data))
            errors.extend(InputValidator._validate_categorical_fields(input_data))
            
            # Business logic validation
            errors.extend(InputValidator._validate_business_rules(input_data))
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info("Input validation passed successfully")
            else:
                logger.warning(f"Input validation failed with {len(errors)} errors")
                
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, [f"Validation system error: {str(e)}"]
    
    @staticmethod
    def _validate_required_fields(input_data: Dict[str, Any]) -> List[str]:
        """Validate that all required fields are present"""
        errors = []
        required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: '{field}'")
            elif input_data[field] is None:
                errors.append(f"Field '{field}' cannot be null")
            elif isinstance(input_data[field], str) and input_data[field].strip() == "":
                errors.append(f"Field '{field}' cannot be empty")
        
        return errors
    
    @staticmethod
    def _validate_numerical_fields(input_data: Dict[str, Any]) -> List[str]:
        """Validate numerical fields with enhanced range checking"""
        errors = []
        
        # Age validation
        if 'age' in input_data:
            age_errors = InputValidator._validate_age(input_data['age'])
            errors.extend(age_errors)
        
        # BMI validation
        if 'bmi' in input_data:
            bmi_errors = InputValidator._validate_bmi(input_data['bmi'])
            errors.extend(bmi_errors)
        
        # Children validation
        if 'children' in input_data:
            children_errors = InputValidator._validate_children(input_data['children'])
            errors.extend(children_errors)
        
        return errors
    
    @staticmethod
    def _validate_age(age_value: Any) -> List[str]:
        """Comprehensive age validation"""
        errors = []
        
        try:
            # Type conversion
            if isinstance(age_value, str):
                age_value = age_value.strip()
                if not age_value.isdigit():
                    errors.append("Age must be a valid integer")
                    return errors
                age_value = int(age_value)
            elif isinstance(age_value, float):
                if age_value != int(age_value):
                    errors.append("Age must be a whole number")
                    return errors
                age_value = int(age_value)
            elif not isinstance(age_value, int):
                errors.append("Age must be a number")
                return errors
            
            # Range validation
            min_age, max_age = Config.VALIDATION_RULES['age']
            if age_value < min_age:
                errors.append(f"Age must be at least {min_age} years")
            elif age_value > max_age:
                errors.append(f"Age cannot exceed {max_age} years")
            
            # Logical validation
            if age_value < 0:
                errors.append("Age cannot be negative")
            
        except ValueError:
            errors.append("Age must be a valid number")
        except Exception as e:
            errors.append(f"Age validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def _validate_bmi(bmi_value: Any) -> List[str]:
        """Comprehensive BMI validation"""
        errors = []
        
        try:
            # Type conversion
            if isinstance(bmi_value, str):
                bmi_value = bmi_value.strip()
                bmi_value = float(bmi_value)
            elif not isinstance(bmi_value, (int, float)):
                errors.append("BMI must be a number")
                return errors
            
            bmi_value = float(bmi_value)
            
            # Range validation
            min_bmi, max_bmi = Config.VALIDATION_RULES['bmi']
            if bmi_value < min_bmi:
                errors.append(f"BMI must be at least {min_bmi}")
            elif bmi_value > max_bmi:
                errors.append(f"BMI cannot exceed {max_bmi}")
            
            # Logical validation
            if bmi_value <= 0:
                errors.append("BMI must be a positive number")
            
        except ValueError:
            errors.append("BMI must be a valid decimal number")
        except Exception as e:
            errors.append(f"BMI validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def _validate_children(children_value: Any) -> List[str]:
        """Comprehensive children count validation"""
        errors = []
        
        try:
            # Type conversion
            if isinstance(children_value, str):
                children_value = children_value.strip()
                if not children_value.isdigit():
                    errors.append("Number of children must be a valid integer")
                    return errors
                children_value = int(children_value)
            elif isinstance(children_value, float):
                if children_value != int(children_value):
                    errors.append("Number of children must be a whole number")
                    return errors
                children_value = int(children_value)
            elif not isinstance(children_value, int):
                errors.append("Number of children must be a number")
                return errors
            
            # Range validation
            min_children, max_children = Config.VALIDATION_RULES['children']
            if children_value < min_children:
                errors.append(f"Number of children cannot be negative")
            elif children_value > max_children:
                errors.append(f"Number of children cannot exceed {max_children}")
            
        except ValueError:
            errors.append("Number of children must be a valid integer")
        except Exception as e:
            errors.append(f"Children validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def _validate_categorical_fields(input_data: Dict[str, Any]) -> List[str]:
        """Validate categorical fields with enhanced checking"""
        errors = []
        
        # Gender validation
        if 'sex' in input_data:
            sex_errors = InputValidator._validate_gender(input_data['sex'])
            errors.extend(sex_errors)
        
        # Smoker validation
        if 'smoker' in input_data:
            smoker_errors = InputValidator._validate_smoker(input_data['smoker'])
            errors.extend(smoker_errors)
        
        # Region validation
        if 'region' in input_data:
            region_errors = InputValidator._validate_region(input_data['region'])
            errors.extend(region_errors)
        
        return errors
    
    @staticmethod
    def _validate_gender(gender_value: Any) -> List[str]:
        """Comprehensive gender validation"""
        errors = []
        
        try:
            if not isinstance(gender_value, str):
                errors.append("Gender must be a text value")
                return errors
            
            gender_value = gender_value.strip().lower()
            valid_genders = [g.lower() for g in Config.VALIDATION_RULES['sex']]
            
            if gender_value not in valid_genders:
                valid_options = ', '.join(Config.VALIDATION_RULES['sex'])
                errors.append(f"Gender must be one of: {valid_options}")
        
        except Exception as e:
            errors.append(f"Gender validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def _validate_smoker(smoker_value: Any) -> List[str]:
        """Comprehensive smoker status validation"""
        errors = []
        
        try:
            if not isinstance(smoker_value, str):
                errors.append("Smoker status must be a text value")
                return errors
            
            smoker_value = smoker_value.strip().lower()
            valid_smoker = [s.lower() for s in Config.VALIDATION_RULES['smoker']]
            
            if smoker_value not in valid_smoker:
                valid_options = ', '.join(Config.VALIDATION_RULES['smoker'])
                errors.append(f"Smoker status must be one of: {valid_options}")
        
        except Exception as e:
            errors.append(f"Smoker validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def _validate_region(region_value: Any) -> List[str]:
        """Comprehensive region validation"""
        errors = []
        
        try:
            if not isinstance(region_value, str):
                errors.append("Region must be a text value")
                return errors
            
            region_value = region_value.strip().lower()
            valid_regions = [r.lower() for r in Config.VALIDATION_RULES['region']]
            
            if region_value not in valid_regions:
                valid_options = ', '.join(Config.VALIDATION_RULES['region'])
                errors.append(f"Region must be one of: {valid_options}")
        
        except Exception as e:
            errors.append(f"Region validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def _validate_business_rules(input_data: Dict[str, Any]) -> List[str]:
        """Validate business logic rules"""
        errors = []
        warnings_list = []
        
        try:
            age = input_data.get('age')
            bmi = input_data.get('bmi')
            children = input_data.get('children')
            smoker = input_data.get('smoker', '').lower()
            
            # Convert values for validation
            if isinstance(age, str):
                age = int(age.strip())
            if isinstance(bmi, str):
                bmi = float(bmi.strip())
            if isinstance(children, str):
                children = int(children.strip())
            
            # Business rule: Children count vs age appropriateness
            if age < 18 and children > 0:
                errors.append("Person under 18 cannot have children dependents")
            
            # Log warnings
            for warning in warnings_list:
                logger.warning(f"Business rule warning: {warning}")
        
        except Exception as e:
            logger.error(f"Business rule validation error: {str(e)}")
        
        return errors
