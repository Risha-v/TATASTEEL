import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class ChartGenerator:
    @staticmethod
    def plot_premium_distribution(data: pd.DataFrame, target: str = "charges"):
        """Plot premium distribution histogram"""
        fig = px.histogram(
            data, 
            x=target, 
            nbins=40, 
            title="Premium Distribution",
            labels={target: "Premium Amount ($)", 'count': 'Frequency'}
        )
        fig.update_layout(
            xaxis_title="Premium Amount ($)",
            yaxis_title="Frequency",
            showlegend=False
        )
        return fig

    @staticmethod
    def plot_feature_importance(importance, feature_names):
        """Plot feature importance"""
        fig = px.bar(
            x=importance,
            y=feature_names,
            orientation='h',
            title="Top Feature Importances",
            labels={'x': 'Importance Score', 'y': 'Features'}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame):
        """Plot correlation matrix"""
        numerical_cols = df.select_dtypes(include=['number']).columns
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix, 
            text_auto=True, 
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        return fig

    @staticmethod
    def plot_model_comparison(metrics_df: pd.DataFrame):
        """Plot model comparison"""
        fig = px.bar(
            metrics_df.reset_index(), 
            x='index', 
            y='test_r2',
            title='Model R² Score Comparison',
            labels={'index': 'Model', 'test_r2': 'R² Score'}
        )
        return fig

    @staticmethod
    def plot_premium_by_factors(data: pd.DataFrame):
        """Plot premium by various factors"""
        fig = px.box(
            data, 
            x='smoker', 
            y='charges', 
            color='sex',
            title="Premium Distribution by Smoking Status and Gender"
        )
        return fig
