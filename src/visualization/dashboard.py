import streamlit as st
from .charts import ChartGenerator

class DashboardComponents:
    @staticmethod
    def overview_metrics(data):
        """Display overview metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Mean Premium", f"${data['charges'].mean():,.2f}")
        with col3:
            smoker_pct = (data['smoker'] == 'yes').mean() * 100
            st.metric("Smokers", f"{smoker_pct:.1f}%")
        with col4:
            st.metric("Avg Age", f"{data['age'].mean():.1f}")

    @staticmethod
    def show_distributions(data):
        """Show premium distributions"""
        chart_gen = ChartGenerator()
        fig = chart_gen.plot_premium_distribution(data)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def show_feature_importance(importance, names):
        """Show feature importance"""
        if importance is not None and names is not None:
            chart_gen = ChartGenerator()
            fig = chart_gen.plot_feature_importance(importance, names)
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def show_correlations(data):
        """Show correlations"""
        chart_gen = ChartGenerator()
        fig = chart_gen.plot_correlation_matrix(data)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def show_premium_factors(data):
        """Show premium by factors"""
        chart_gen = ChartGenerator()
        fig = chart_gen.plot_premium_by_factors(data)
        st.plotly_chart(fig, use_container_width=True)
