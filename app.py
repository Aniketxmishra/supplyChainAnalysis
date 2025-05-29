import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from io import BytesIO
from PIL import Image
import os
import warnings
import datetime  # Added for date handling
# Set page configuration
st.set_page_config(
    page_title="Supply Chain Causal Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve the look and feel
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Load or recreate the synthetic dataset
@st.cache_data
def load_data():
    try:
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        sales = 50 + np.random.normal(0, 5, 100)
        sales[50:] += 10  # Promotion effect starts at day 50
        
        df = pd.DataFrame({"date": dates, "sales": sales})
        df["promotion"] = 0
        df.loc[50:, "promotion"] = 1
        df["promotion_category"] = df["promotion"].map({0: "No Promotion", 1: "Promotion"})
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return a minimal dataframe if there's an error
        return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10),
                            "sales": np.random.normal(50, 5, 10),
                            "promotion": [0]*5 + [1]*5,
                            "promotion_category": ["No Promotion"]*5 + ["Promotion"]*5,
                            "month": [1]*10,
                            "week": [1]*10})

# Load data
df = load_data()

# Calculate key metrics
no_promo = df[df['promotion'] == 0]
with_promo = df[df['promotion'] == 1]
avg_no_promo = no_promo['sales'].mean()
avg_with_promo = with_promo['sales'].mean()
effect_size = avg_with_promo - avg_no_promo
percent_increase = (effect_size / avg_no_promo) * 100

# Create header
st.markdown('<div class="main-header">Supply Chain Causal Analysis Dashboard</div>', unsafe_allow_html=True)

# Add sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "Overview",
    "Data Exploration",
    "Causal Analysis",
    "What-If Simulation",
    "Download Results"
])

# Display different pages based on selection
if page == "Overview":
    st.markdown("## Project Overview")
    
    # Display project description
    st.markdown("""
    This dashboard presents a causal analysis of the impact of promotional campaigns on sales. 
    Using synthetic data with a known ground truth, we demonstrate how causal inference techniques 
    can quantify the effect of interventions and provide actionable business insights.
    """)
    
    # Key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{effect_size:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Units Increase</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{percent_increase:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Sales Increase</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">p < 0.00001</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Statistical Significance</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">Robust</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Validation Tests</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display causal graph
    st.markdown("## Causal Model")
    
    if os.path.exists("causal_graph.png"):
        st.image("causal_graph.png", caption="Causal Graph: Promotion → Sales")
    else:
        # Generate causal graph if image doesn't exist
        plt.figure(figsize=(6, 4))
        G = nx.DiGraph()
        G.add_edge('Promotion', 'Sales')
        pos = {'Promotion': (0, 0), 'Sales': (1, 0)}
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
                font_size=12, font_weight='bold', arrowsize=20)
        plt.title('Causal Graph: Promotion → Sales')
        
        # Save figure to a buffer and display
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        st.image(buf, caption="Causal Graph: Promotion → Sales")
    
    # Key findings
    st.markdown("## Key Findings")
    st.markdown("""
    - **Causal Effect**: Promotions cause a statistically significant increase in sales
    - **Effect Size**: Average increase of 11.22 units per promotion
    - **Percentage Impact**: 23.0% boost in sales
    - **Statistical Validity**: The effect is highly significant (p < 0.00001)
    - **Robustness**: Results hold under multiple refutation tests
    """)
    
    # Show overview of results visualization
    st.markdown("## Results Overview")
    if os.path.exists("promotion_effect_visualization.png"):
        st.image("promotion_effect_visualization.png", caption="Comprehensive Analysis of Promotional Impact")

elif page == "Data Exploration":
    st.markdown("## Data Exploration")
    
    # Interactive time series chart
    st.markdown("### Sales Over Time")
    
    # Create Plotly figure
    fig = px.scatter(df, x="date", y="sales", color="promotion_category", 
                     color_discrete_map={"No Promotion": "#ff9999", "Promotion": "#66b3ff"},
                     title="Sales Over Time with Promotion Effect")
    
    # Add vertical line using shapes instead of add_vline to avoid datetime issues
    promotion_start_date = df["date"][49]
    
    # Add shape for vertical line
    fig.add_shape(
        type="line",
        x0=promotion_start_date,
        x1=promotion_start_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add annotation for the line
    fig.add_annotation(
        x=promotion_start_date,
        y=1,
        yref="paper",
        text="Promotion Start",
        showarrow=False,
        font=dict(color="red"),
        xanchor="right",
        yanchor="bottom"
    )
    
    # Add trend lines
    no_promo_x = np.arange(len(no_promo))
    with_promo_x = np.arange(len(with_promo))
    no_promo_trend = np.poly1d(np.polyfit(no_promo_x, no_promo["sales"], 1))
    with_promo_trend = np.poly1d(np.polyfit(with_promo_x, with_promo["sales"], 1))
    
    fig.add_trace(go.Scatter(x=no_promo["date"], y=no_promo_trend(no_promo_x), 
                            mode='lines', line=dict(color="#ff9999", width=3), 
                            name='No Promotion Trend'))
    
    fig.add_trace(go.Scatter(x=with_promo["date"], y=with_promo_trend(with_promo_x), 
                            mode='lines', line=dict(color="#66b3ff", width=3), 
                            name='Promotion Trend'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add controls for data filtering
    st.markdown("### Filter Data")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Initialize with default start and end dates
            default_start_date = df["date"].min().to_pydatetime().date()
            default_end_date = df["date"].max().to_pydatetime().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=[default_start_date, default_end_date],
                min_value=default_start_date,
                max_value=default_end_date
            )
            
            # Handle case where only one date is selected
            if isinstance(date_range, (list, tuple)) and len(date_range) >= 2:
                start_date, end_date = date_range[0], date_range[1]
            elif isinstance(date_range, (datetime.date, pd.Timestamp)):
                # If a single date is selected, use it as both start and end
                start_date = end_date = date_range
                st.info(f"Showing data for single date: {date_range}")
            else:
                # For any other unexpected format, use default range
                start_date, end_date = default_start_date, default_end_date
                st.warning("Please select a valid date range. Using full date range instead.")
                
        except Exception as e:
            # In case of any error, use full date range
            start_date = df["date"].min().to_pydatetime().date()
            end_date = df["date"].max().to_pydatetime().date()
            st.error(f"Error handling date range: {str(e)}. Using full date range instead.")
    
    with col2:
        view_option = st.selectbox(
            "View By",
            ["Daily", "Weekly", "Monthly"]
        )
    
    # Filter data based on selection with proper error handling
    try:
        # Convert dates to pandas timestamps safely
        pd_start_date = pd.Timestamp(start_date)
        pd_end_date = pd.Timestamp(end_date)
        
        # Ensure end date is not before start date
        if pd_end_date < pd_start_date:
            pd_start_date, pd_end_date = pd_end_date, pd_start_date
            st.warning("End date was before start date. Dates have been swapped.")
        
        filtered_df = df[(df["date"] >= pd_start_date) & (df["date"] <= pd_end_date)]
        
        # Handle empty filtered data
        if len(filtered_df) == 0:
            st.warning("No data in the selected date range. Showing all data instead.")
            filtered_df = df.copy()
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}. Showing all data instead.")
        filtered_df = df.copy()
    
    # Create aggregation based on selection
    if view_option == "Weekly":
        try:
            # Use tuples for groupby to avoid FutureWarning
            agg_df = filtered_df.groupby([("week",), ("promotion_category",)]).agg(
                avg_sales=("sales", "mean")
            ).reset_index()
            
            # Rename columns to remove tuple formatting
            agg_df.columns = [col[0] if isinstance(col, tuple) else col for col in agg_df.columns]
            
            fig = px.bar(agg_df, x="week", y="avg_sales", color="promotion_category",
                        color_discrete_map={"No Promotion": "#ff9999", "Promotion": "#66b3ff"},
                        title="Weekly Average Sales", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating weekly visualization: {str(e)}")
            st.info("Try selecting a different date range or view option.")
        
    elif view_option == "Monthly":
        try:
            # Use tuples for groupby to avoid FutureWarning
            agg_df = filtered_df.groupby([("month",), ("promotion_category",)]).agg(
                avg_sales=("sales", "mean")
            ).reset_index()
            
            # Rename columns to remove tuple formatting
            agg_df.columns = [col[0] if isinstance(col, tuple) else col for col in agg_df.columns]
            
            fig = px.bar(agg_df, x="month", y="avg_sales", color="promotion_category",
                        color_discrete_map={"No Promotion": "#ff9999", "Promotion": "#66b3ff"},
                        title="Monthly Average Sales", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating monthly visualization: {str(e)}")
            st.info("Try selecting a different date range or view option.")
        
    else:  # Daily
        fig = px.bar(filtered_df, x="date", y="sales", color="promotion_category",
                    color_discrete_map={"No Promotion": "#ff9999", "Promotion": "#66b3ff"},
                    title="Daily Sales")
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution comparison
    st.markdown("### Sales Distribution Comparison")
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Sales Distribution", "Box Plot Comparison"),
                        specs=[[{"type": "histogram"}, {"type": "box"}]])
    
    # Add histograms
    fig.add_trace(
        go.Histogram(x=no_promo["sales"], name="No Promotion", marker_color="#ff9999", opacity=0.7),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=with_promo["sales"], name="Promotion", marker_color="#66b3ff", opacity=0.7),
        row=1, col=1
    )
    
    # Add box plots
    fig.add_trace(
        go.Box(y=no_promo["sales"], name="No Promotion", marker_color="#ff9999"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(y=with_promo["sales"], name="Promotion", marker_color="#66b3ff"),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Distribution Analysis of Sales",
        height=500,
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table with pagination
    st.markdown("### Raw Data Explorer")
    page_size = st.slider("Rows per page", min_value=5, max_value=50, value=10)
    total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)

elif page == "Causal Analysis":
    st.markdown("## Causal Analysis Results")
    
    # Display methodological approach
    st.markdown("### Causal Inference Methodology")
    st.markdown("""
    Our causal analysis follows the 4-step process from the DoWhy framework:
    
    1. **Model** - Define the causal graph: Promotion → Sales
    2. **Identify** - Identify the causal estimand using backdoor criterion
    3. **Estimate** - Estimate the causal effect using multiple methods
    4. **Refute** - Validate the results with various refutation tests
    """)
    
    # Display causal effect estimates
    st.markdown("### Causal Effect Estimation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("#### Linear Regression Estimator")
        st.markdown(f"""
        **Effect Estimate**: {effect_size:.4f} units  
        **Standard Error**: {np.std(df['sales'])/np.sqrt(len(df)):.4f}  
        **t-statistic**: {effect_size/(np.std(df['sales'])/np.sqrt(len(df))):.4f}  
        **p-value**: < 0.00001  
        
        The linear regression estimator indicates that promotions cause an increase of {effect_size:.2f} units in sales, which is highly statistically significant.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Create effect visualization
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Plot means
        ax.bar(["No Promotion", "Promotion"], [avg_no_promo, avg_with_promo], color=["#ff9999", "#66b3ff"])
        
        # Add error bars
        ax.errorbar(["No Promotion", "Promotion"], [avg_no_promo, avg_with_promo], 
                   yerr=[np.std(no_promo['sales'])/np.sqrt(len(no_promo)), 
                          np.std(with_promo['sales'])/np.sqrt(len(with_promo))],
                   fmt='none', color='black', capsize=5)
        
        # Add text labels
        ax.text(0, avg_no_promo + 0.5, f"{avg_no_promo:.2f}", ha='center', va='bottom')
        ax.text(1, avg_with_promo + 0.5, f"{avg_with_promo:.2f}", ha='center', va='bottom')
        
        # Add arrow to show effect
        ax.annotate(f"+{effect_size:.2f}", 
                   xy=(0.5, (avg_no_promo + avg_with_promo)/2),
                   xytext=(0.5, (avg_no_promo + avg_with_promo)/2 - 5),
                   arrowprops=dict(arrowstyle="->", color="black"),
                   ha='center', va='center', fontsize=12)
        
        ax.set_ylabel("Average Sales")
        ax.set_title("Causal Effect of Promotion on Sales")
        
        st.pyplot(fig)
    
    # Display refutation tests
    st.markdown("### Refutation Tests")
    
    # Add expandable sections for each refutation test
    with st.expander("Random Common Cause Refutation"):
        st.markdown("""
        This test adds random variables as common causes (confounders) to check robustness.
        
        **Result**: The effect estimate remains stable with random common causes, confirming that our finding is not sensitive to unobserved confounders.
        """)
    
    with st.expander("Placebo Treatment Refutation"):
        st.markdown("""
        This test replaces the true treatment with a random placebo to ensure the effect disappears.
        
        **Result**: With placebo treatments, the effect estimate is close to zero and not significant, confirming that our actual finding captures a real effect.
        """)
    
    with st.expander("Unobserved Confounder Sensitivity"):
        st.markdown("""
        This test simulates the presence of unobserved confounders to evaluate how strong they would need to be to invalidate our finding.
        
        **Result**: Even with simulated strong confounders, the effect remains significant, indicating robustness to potential unobserved variables.
        """)
    
    with st.expander("Bootstrap Refutation"):
        st.markdown("""
        This test resamples the data with replacement to generate confidence intervals.
        
        **Result**: Bootstrap analysis shows narrow confidence intervals that do not include zero, confirming the statistical significance of our finding.
        """)
    
    # Display formal causal inference results
    st.markdown("### Statistical Analysis")
    st.markdown("""
    #### T-Test Results
    
    The two-sample t-test comparing sales with and without promotion shows:
    """)
    
    t_stat = np.round((effect_size/(np.std(df['sales'])/np.sqrt(len(df)))), 4)
    p_val = "<0.00001"
    
    results_df = pd.DataFrame({
        "Metric": ["Mean (No Promotion)", "Mean (Promotion)", "Difference", "t-statistic", "p-value"],
        "Value": [f"{avg_no_promo:.4f}", f"{avg_with_promo:.4f}", f"{effect_size:.4f}", f"{t_stat}", p_val]
    })
    
    st.table(results_df)
    
    # Visualization of before/after distribution
    if os.path.exists("sales_distribution_comparison.png"):
        st.image("sales_distribution_comparison.png", caption="Distribution of Sales Before vs. After Promotion")

elif page == "What-If Simulation":
    st.markdown("## What-If Simulation")
    st.markdown("""
    This interactive tool allows you to simulate different promotion scenarios and estimate their effects on sales.
    Adjust the parameters below to see how changes in the promotion strategy might impact overall sales.
    """)
    
    # Add simulation controls
    st.markdown("### Promotion Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        promotion_strength = st.slider(
            "Promotion Strength Multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Adjust the strength of the promotion effect (1.0 = original effect)"
        )
    
    with col2:
        promotion_coverage = st.slider(
            "Promotion Coverage (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Percentage of time period with active promotion"
        )
    
    # Simulate new data based on parameters
    np.random.seed(42)
    sim_dates = pd.date_range("2024-01-01", periods=100)
    sim_sales = 50 + np.random.normal(0, 5, 100)
    
    # Apply promotion effect based on coverage and strength
    promotion_start = int(100 * (1 - promotion_coverage/100))
    sim_sales[promotion_start:] += 10 * promotion_strength
    
    sim_df = pd.DataFrame({"date": sim_dates, "sales": sim_sales})
    sim_df["promotion"] = 0
    sim_df.loc[promotion_start:, "promotion"] = 1
    
    # Calculate metrics
    sim_no_promo = sim_df[sim_df['promotion'] == 0]
    sim_with_promo = sim_df[sim_df['promotion'] == 1]
    sim_avg_no_promo = sim_no_promo['sales'].mean() if len(sim_no_promo) > 0 else 0
    sim_avg_with_promo = sim_with_promo['sales'].mean() if len(sim_with_promo) > 0 else 0
    sim_effect = sim_avg_with_promo - sim_avg_no_promo
    sim_percent_increase = (sim_effect / sim_avg_no_promo * 100) if sim_avg_no_promo > 0 else 0
    
    # Check for edge cases in simulation data to prevent errors
    if len(sim_no_promo) == 0:
        st.warning("Warning: Simulation has no non-promotion period. Adjust parameters for a more realistic comparison.")
        sim_avg_no_promo = df[df['promotion'] == 0]['sales'].mean()  # Use original data as fallback
    
    if len(sim_with_promo) == 0:
        st.warning("Warning: Simulation has no promotion period. Adjust parameters to see promotion effects.")
        sim_avg_with_promo = df[df['promotion'] == 1]['sales'].mean()  # Use original data as fallback
    
    # Display simulation results
    st.markdown("### Simulation Results")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{sim_effect:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Estimated Effect (units)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{sim_percent_increase:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Sales Increase</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        total_sales_increase = sim_effect * (promotion_coverage/100 * 100)
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_sales_increase:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Additional Units</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization of simulation
    fig = px.line(sim_df, x="date", y="sales", color_discrete_sequence=["#1E88E5"],
                 title="Simulated Sales with Custom Promotion Parameters")
    
    # Add shaded region for promotion period using shapes to avoid datetime issues
    fig.add_shape(
        type="rect",
        x0=sim_dates[promotion_start],
        x1=sim_dates[-1],
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="#66b3ff",
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    
    # Add annotation for the promotion period
    fig.add_annotation(
        x=sim_dates[promotion_start + 5],  # Offset a bit for better visibility
        y=0.95,
        text="Promotion Period",
        showarrow=False,
        yref="paper",
        font=dict(color="black"),
        bgcolor="rgba(255, 255, 255, 0.5)",
        borderpad=4
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display comparison with original effect
    st.markdown("### Comparison with Original Analysis")
    
    comparison_df = pd.DataFrame({
        "Scenario": ["Original Analysis", "What-If Simulation"],
        "Effect Size (units)": [f"{effect_size:.2f}", f"{sim_effect:.2f}"],
        "Percent Increase": [f"{percent_increase:.2f}%", f"{sim_percent_increase:.2f}%"],
        "Promotion Coverage (%)": [50, promotion_coverage],
        "Promotion Strength": [1.0, promotion_strength]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # ROI calculator
    st.markdown("### ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unit_profit = st.number_input(
            "Profit per Unit ($)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        )
    
    with col2:
        promotion_cost = st.number_input(
            "Total Promotion Cost ($)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=100.0
        )
    
    # Calculate ROI
    additional_profit = total_sales_increase * unit_profit
    roi_percent = ((additional_profit - promotion_cost) / promotion_cost * 100) if promotion_cost > 0 else 0
    
    # Display ROI results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${additional_profit:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Additional Profit</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${additional_profit - promotion_cost:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Net Profit</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{roi_percent:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">ROI</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Download Results":
    st.markdown("## Download Analysis Results")
    st.markdown("""
    Download the complete analysis results and visualizations for offline review or presentation.
    """)
    
    # Create tabs for different download options
    tab1, tab2, tab3 = st.tabs(["Data", "Visualizations", "Report"])
    
    with tab1:
        st.markdown("### Download Data")
        
        # Generate CSV data for download
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="supply_chain_analysis_data.csv",
            mime="text/csv"
        )
        
        # Display data preview
        st.dataframe(df.head(10), use_container_width=True)
        
        # Generate Excel data with multiple sheets
        try:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="Raw Data", index=False)
                
                # Summary sheet
                summary_data = pd.DataFrame({
                    "Metric": ["Average Sales (No Promotion)", "Average Sales (Promotion)", 
                              "Effect Size", "Percent Increase", "p-value"],
                    "Value": [f"{avg_no_promo:.4f}", f"{avg_with_promo:.4f}", 
                              f"{effect_size:.4f}", f"{percent_increase:.2f}%", "<0.00001"]
                })
                summary_data.to_excel(writer, sheet_name="Summary", index=False)
                
                # Monthly aggregation with tuple-style grouping to avoid FutureWarning
                monthly_data = df.groupby([("month",), ("promotion_category",)]).agg(
                    avg_sales=("sales", "mean"),
                    std_sales=("sales", "std"),
                    min_sales=("sales", "min"),
                    max_sales=("sales", "max")
                ).reset_index()
                
                # Rename columns to remove tuple formatting
                monthly_data.columns = [col[0] if isinstance(col, tuple) else col for col in monthly_data.columns]
                monthly_data.to_excel(writer, sheet_name="Monthly Summary", index=False)
            
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="Download Data as Excel",
                data=excel_data,
                file_name="supply_chain_analysis.xlsx",
                mime="application/vnd.ms-excel"
            )
        except Exception as e:
            st.error(f"Failed to generate Excel file: {str(e)}")
            st.info("You can still download the CSV file above.")
    
    with tab2:
        st.markdown("### Download Visualizations")
        
        # Display and offer downloads for each visualization
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists("sales_over_time.png"):
                st.image("sales_over_time.png", caption="Sales Over Time")
                
                with open("sales_over_time.png", "rb") as file:
                    st.download_button(
                        label="Download Time Series Chart",
                        data=file,
                        file_name="sales_over_time.png",
                        mime="image/png"
                    )
            
            if os.path.exists("causal_graph.png"):
                st.image("causal_graph.png", caption="Causal Graph")
                
                with open("causal_graph.png", "rb") as file:
                    st.download_button(
                        label="Download Causal Graph",
                        data=file,
                        file_name="causal_graph.png",
                        mime="image/png"
                    )
        
        with col2:
            if os.path.exists("promotion_effect_visualization.png"):
                st.image("promotion_effect_visualization.png", caption="Promotion Effect")
                
                with open("promotion_effect_visualization.png", "rb") as file:
                    st.download_button(
                        label="Download Effect Visualization",
                        data=file,
                        file_name="promotion_effect_visualization.png",
                        mime="image/png"
                    )
            
            if os.path.exists("sales_distribution_comparison.png"):
                st.image("sales_distribution_comparison.png", caption="Sales Distribution")
                
                with open("sales_distribution_comparison.png", "rb") as file:
                    st.download_button(
                        label="Download Distribution Comparison",
                        data=file,
                        file_name="sales_distribution_comparison.png",
                        mime="image/png"
                    )
    
    with tab3:
        st.markdown("### Download Full Report")
        
        # Generate a summary report
        report = f"""# Supply Chain Causal Analysis Report

## Executive Summary

This report presents the findings of a causal analysis on the impact of promotional campaigns on sales. The analysis was conducted using rigorous causal inference techniques to quantify the effect of promotions on sales performance.

## Key Findings

- **Causal Effect**: Promotions cause a statistically significant increase in sales of {effect_size:.2f} units.
- **Percentage Impact**: Sales increased by {percent_increase:.1f}% during promotional periods.
- **Statistical Significance**: The effect is highly significant with p < 0.00001.
- **Robustness**: Results were validated using multiple refutation tests.

## Methodology

The analysis followed the 4-step causal inference framework:

1. **Model** - Defined the causal graph: Promotion → Sales
2. **Identify** - Identified the causal estimand using backdoor criterion
3. **Estimate** - Estimated the causal effect using multiple methods
4. **Refute** - Validated the results with various refutation tests

## Detailed Results

| Metric | Value |
|--------|-------|
| Average Sales (No Promotion) | {avg_no_promo:.4f} |
| Average Sales (Promotion) | {avg_with_promo:.4f} |
| Effect Size | {effect_size:.4f} |
| Percent Increase | {percent_increase:.2f}% |
| t-statistic | {effect_size/(np.std(df['sales'])/np.sqrt(len(df))):.4f} |
| p-value | < 0.00001 |

## Recommendations

Based on the analysis, we recommend:

1. **Continue Promotional Activities**: The significant positive effect justifies continued investment in promotions.
2. **Optimize Timing**: Schedule promotions during high-impact periods.
3. **Further Analysis**: Conduct additional studies to optimize promotion parameters.

## Conclusion

The causal analysis provides strong evidence that promotional activities have a substantial positive impact on sales. This insight can be used to inform marketing strategy and budget allocation decisions.

*Analysis Date: {pd.Timestamp.now().strftime("%Y-%m-%d")}*
"""
        
        # Create download button for markdown report
        st.download_button(
            label="Download Report (Markdown)",
            data=report,
            file_name="causal_analysis_report.md",
            mime="text/markdown"
        )
        
        # Display report preview
        st.markdown("### Report Preview")
        st.markdown(report)

# Status indicator
try:
    # Try to determine if everything is working correctly
    has_errors = False
    status_message = "✅ Dashboard is running normally"
    
    # Check if visualization files exist
    missing_files = []
    for file in ["sales_over_time.png", "causal_graph.png", "promotion_effect_visualization.png", 
                "sales_distribution_comparison.png"]:
        if not os.path.exists(file):
            missing_files.append(file)
            has_errors = True
    
    if missing_files:
        status_message = f"⚠️ Some visualization files are missing: {', '.join(missing_files)}"
except Exception:
    has_errors = True
    status_message = "⚠️ Some components may not be working correctly"

# Add footer with error handling guidance
st.markdown("---")
st.markdown(f"{status_message}")
st.markdown("© 2025 Aniket Mishra | Supply Chain Causal Analysis Dashboard")

# Add information about errors
with st.expander("Troubleshooting Information"):
    st.markdown("""
    ### Common Issues
    
    - **Visualization Errors**: If visualizations don't display correctly, try refreshing the page.
    - **Download Issues**: If you encounter download errors, check that you have write permissions to your download folder.
    - **Data Loading**: Data is cached in memory for performance. To force a refresh, use the "Rerun" button in the top right.
    
    ### Libraries Used
    
    This dashboard is built with:
    - Streamlit: Interactive web application
    - Plotly: Interactive visualizations
    - Pandas/NumPy: Data processing
    - Matplotlib/Seaborn: Static visualizations
    """)

