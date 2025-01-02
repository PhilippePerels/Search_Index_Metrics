import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

data_path = "agg_year_month_completed.csv"
df = pd.read_csv(data_path)

# Convert year_month to datetime
df['year_month'] = pd.to_datetime(df['year_month'], errors='coerce')

# Metrics normalized for search index calculation
metrics_for_index = {
    'global_search_volume_norm': 0.05,
    'market_share_norm': 0.05,
    'yoy_growth_norm': 0.20,
    'qoq_growth_norm': 0.18,
    'mom_growth_norm': 0.18,
    'contribution_cat_growth_norm': 0.10,
    'seasonality_index_norm': 0.06,
    'geo_diversity_norm': 0.06,
    'peer_comparison_score_norm': 0.12
}

# Convert needed metrics to numeric and fill NaN
for m in metrics_for_index.keys():
    df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)

# Define metrics
metrics = [
    'global_search_volume', 'segment_total_volume', 'market_share', 
    'segment_avg_volume', 'peer_comparison_score', 'qoq_growth', 
    'segment_cat_growth_rate', 'geo_diversity', 'seasonality_index',
    'yoy_growth', 'mom_growth', 'contribution_cat_growth', 'search_index'
]

# Metric descriptions
metric_descriptions = {
    'global_search_volume': "Total number of searches for the brand.",
    'segment_total_volume': "Total search volume for all brands in the selected segment(s).",
    'market_share': "Brand's percentage share of the segment's total search volume.",
    'segment_avg_volume': "Average search volume of brands in the segment.",
    'peer_comparison_score': "Brand volume vs. peer average, scaled by 100. Above 100 means above-average interest.",
    'qoq_growth': "Quarter-over-quarter growth in search volume.",
    'segment_cat_growth_rate': "Segment-level category growth rate year-over-year.",
    'geo_diversity': "Diversity of search volume across different geographies.",
    'seasonality_index': "Index reflecting seasonality of search interest.",
    'yoy_growth': "Year-over-year growth in search volume.",
    'mom_growth': "Month-over-month growth in search volume.",
    'contribution_cat_growth': "Brand's contribution to overall category growth.",
    'search_index': "Composite index calculated from normalized metrics based on chosen weights."
}

# Segment-level metrics list (for ignoring brand filter)
segment_level_metrics = ['segment_total_volume', 'segment_avg_volume', 'segment_cat_growth_rate']

# Title and description
st.title("Search Index Metrics Visualization")
st.markdown("""
This dashboard allows you to visualize various metrics for multiple brands and segments.
You can adjust the weights of the normalized metrics that compose the **search_index** using the sliders in the sidebar.
The search_index will be recalculated dynamically based on the chosen weights.

**Note:** Only a maximum of 12 brands can be selected at once for clarity.
""")

# Sidebar for filters
st.sidebar.header("Filters & Settings")

# Choose segments
all_segments = df['segment'].dropna().unique().tolist()
selected_segments = st.sidebar.multiselect("Select Segments (Multiple):", all_segments, default=[all_segments[0]])

# Filter brands by selected segments
filtered = df[df['segment'].isin(selected_segments)]
all_brands = filtered['brand'].dropna().unique().tolist()

# Select some brands by default (up to 5 if available)
default_brands = all_brands[:5]
selected_brands = st.sidebar.multiselect("Select Brands (Multiple, max 12):", all_brands, default=default_brands)

# Enforce maximum 12 brands
if len(selected_brands) > 12:
    st.sidebar.warning("You have selected more than 12 brands. Only the first 12 will be used.")
    selected_brands = selected_brands[:12]

# Date range selection
min_date = df['year_month'].min()
max_date = df['year_month'].max()
date_range = st.sidebar.date_input("Select Date Range:", value=[min_date, max_date])
if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Adjust weights
st.sidebar.markdown("### Adjust Weights for Search Index")
user_weights = {}
for m, default_w in metrics_for_index.items():
    user_weights[m] = st.sidebar.slider(f"Weight for {m}", 0.0, 0.5, float(default_w), 0.01)

# Normalize weights so that sum = 1
total_weight = sum(user_weights.values())
num_metrics = len(user_weights)
if total_weight > 0:
    normalized_weights = {k: round(v / total_weight, 2) for k, v in user_weights.items()}
else:
    # If all weights are zero, distribute them equally
    equal_weight = round(1.0 / num_metrics, 2)
    normalized_weights = {k: equal_weight for k in user_weights.keys()}

# Recalculate search_index with normalized weights
# Also ensure that if we round, we still sum to 1. We can adjust the last metric to fix rounding issues.
sum_norm = sum(normalized_weights.values())
if sum_norm != 1.0:
    # Adjust last metric to make sure sum is exactly 1.0 after rounding
    diff = 1.0 - sum_norm
    keys_list = list(normalized_weights.keys())
    normalized_weights[keys_list[-1]] = round(normalized_weights[keys_list[-1]] + diff, 2)
    # Re-check sum
    sum_norm = sum(normalized_weights.values())
    if sum_norm != 1.0:
        # If still off due to rounding, we accept a minor deviation or re-round
        pass

df_calc = df.copy()
df_calc['search_index'] = 0
for m, w in normalized_weights.items():
    df_calc['search_index'] += df_calc[m] * w

# Filter data by date and segment
dff = df_calc.copy()
dff = dff[(dff['year_month'] >= pd.to_datetime(start_date)) & (dff['year_month'] <= pd.to_datetime(end_date))]

if dff.empty:
    st.write("No data available for the selected filters.")
else:
    for met in metrics:
        dff[met] = pd.to_numeric(dff[met], errors='coerce')
        
        if met in segment_level_metrics:
            # Segment-level metric: only filter by segment
            df_metric = dff[dff['segment'].isin(selected_segments)].copy()
            df_metric.sort_values(by='year_month', inplace=True)
            if df_metric.empty:
                continue
            fig = px.line(df_metric, x='year_month', y=met, color='segment',
                          title=f"{met} over time for selected {', '.join(selected_segments)}",
                          line_shape='linear')
        else:
            # Brand-level metric: filter by brand and segment
            df_metric = dff[dff['segment'].isin(selected_segments) & dff['brand'].isin(selected_brands)].copy()
            df_metric.sort_values(by='year_month', inplace=True)
            if df_metric.empty:
                continue
            fig = px.line(df_metric, x='year_month', y=met, color='brand',
                          title=f"{met} over time for selected {', '.join(selected_segments)}",
                          line_shape='linear')

        # Prevent connecting gaps between start and end points
        fig.update_traces(connectgaps=False)
        
        fig.update_layout(xaxis_title='Date', yaxis_title=met)
        st.plotly_chart(fig, use_container_width=True)

        # Show description below the graph
        desc = metric_descriptions.get(met, "No description available.")
        st.markdown(f"**Description for {met}:** {desc}")

    # Add a correlation heatmap of the main metrics
    # Compute correlation on numeric columns
    numeric_cols = ['global_search_volume', 'segment_total_volume', 'market_share',
                    'segment_avg_volume', 'peer_comparison_score', 'qoq_growth',
                    'segment_cat_growth_rate', 'geo_diversity', 'seasonality_index',
                    'yoy_growth', 'mom_growth', 'contribution_cat_growth', 'search_index']
    numeric_cols = [c for c in numeric_cols if c in dff.columns]
    corr_matrix = dff[numeric_cols].corr()

    st.markdown("### Correlation Heatmap of Selected Metrics")
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto', color_continuous_scale='RdBu', origin='lower')
    fig_corr.update_layout(title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
