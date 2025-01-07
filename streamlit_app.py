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

# Define metrics (the order here will match the final anchor order in the dashboard)
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

# ----------------------------------------------------------------
# MAIN DASHBOARD
# ----------------------------------------------------------------

# 1) Title and Header
st.title("Search Index Metrics Visualization")
st.markdown("""
This dashboard allows you to visualize various metrics for multiple brands and segments.
You can adjust the weights of the normalized metrics that compose the **search_index** using the sliders in the sidebar.
The search_index will be recalculated dynamically based on the chosen weights.

**Note:** Only a maximum of 15 brands can be selected at once for clarity.
""")

# ----------------------------------------------------------------
# 2) SIDEBAR
# ----------------------------------------------------------------

st.sidebar.header("Filters & Settings")

# -- Table of Contents in One Block --
# Build the metric links in the same order as they will appear in the dashboard
toc_links = ""
for metric in metrics:
    anchor_name = metric.replace("_", "-")
    toc_links += f'<li style="margin-bottom:2px;"><a href="#{anchor_name}">{metric}</a></li>'

# A single markdown block with minimal spacing for each link
st.sidebar.markdown(
    f"""
    ### Table of Contents
    <ul style="margin-bottom: 5px; list-style-position: inside;">
      <li style="margin-bottom:2px;"><a href="#search-index-graph">Search Index Graph</a></li>
      <li style="margin-bottom:2px;"><a href="#yoy-ranking-table">YoY Ranking Table (2023 vs 2024)</a></li>
      {toc_links}
      <li style="margin-bottom:2px;"><a href="#correlation-heatmap">Correlation Heatmap</a></li>
    </ul>
    """,
    unsafe_allow_html=True
)

# Choose segments
all_segments = df['segment'].dropna().unique().tolist()
selected_segments = st.sidebar.multiselect("Select Segments (Multiple):", all_segments, default=[all_segments[0]])

# Filter brands by selected segments
filtered = df[df['segment'].isin(selected_segments)]
all_brands = filtered['brand'].dropna().unique().tolist()
default_brands = all_brands[:5]
selected_brands = st.sidebar.multiselect("Select Brands (Multiple, max 15):", all_brands, default=default_brands)

if len(selected_brands) > 15:
    st.sidebar.warning("You have selected more than 15 brands. Only the first 15 will be used.")
    selected_brands = selected_brands[:15]

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

# Normalize weights
total_weight = sum(user_weights.values())
num_metrics = len(user_weights)
if total_weight > 0:
    normalized_weights = {k: round(v / total_weight, 2) for k, v in user_weights.items()}
else:
    equal_weight = round(1.0 / num_metrics, 2)
    normalized_weights = {k: equal_weight for k in user_weights.keys()}

sum_norm = sum(normalized_weights.values())
if sum_norm != 1.0:
    diff = 1.0 - sum_norm
    keys_list = list(normalized_weights.keys())
    normalized_weights[keys_list[-1]] = round(normalized_weights[keys_list[-1]] + diff, 2)

# Ranking Scope
ranking_scope = st.sidebar.radio(
    "Ranking Scope for YoY Ranking:",
    ["Segment", "Global"],
    help="""
    - **Segment**: The rank is computed among brands filtered by the selected segment(s) and date range.
    - **Global**: The rank is computed among *all* brands in the entire dataset, ignoring segment/date filters.
    """
)

# ----------------------------------------------------------------
# 3) DATA PREPARATION & FILTERING
# ----------------------------------------------------------------

df_calc = df.copy()
df_calc['search_index'] = 0
for m, w in normalized_weights.items():
    df_calc['search_index'] += df_calc[m] * w

dff = df_calc.copy()
dff = dff[(dff['year_month'] >= pd.to_datetime(start_date)) & (dff['year_month'] <= pd.to_datetime(end_date))]
dff = dff[dff['segment'].isin(selected_segments)]

# ----------------------------------------------------------------
# 4) MAIN CONTENT
# ----------------------------------------------------------------

if dff.empty:
    st.write("No data available for the selected filters.")
else:
    # --------------------------------------------------------
    # ANCHOR: #search-index-graph
    # --------------------------------------------------------
    st.markdown('<a name="search-index-graph"></a>', unsafe_allow_html=True)
    st.subheader("Search Index Graph")

    if "search_index" in metrics:
        df_metric = dff[dff['brand'].isin(selected_brands)].copy()
        df_metric.sort_values(by='year_month', inplace=True)

        if not df_metric.empty:
            fig = px.line(
                df_metric,
                x='year_month', y='search_index', color='brand',
                title="search_index over time for selected segments",
                line_shape='linear'
            )
            fig.update_traces(connectgaps=False)
            fig.update_layout(xaxis_title='Date', yaxis_title='search_index')
            st.plotly_chart(fig, use_container_width=True)

        # Remove "search_index" from metrics so we don't re-plot it later
        metrics.remove("search_index")

    # --------------------------------------------------------
    # ANCHOR: #yoy-ranking-table
    # --------------------------------------------------------
    st.markdown('<a name="yoy-ranking-table"></a>', unsafe_allow_html=True)
    st.subheader("YoY Ranking Table (2023 vs 2024)")

    if ranking_scope == "Segment":
        yoy_ranking_df = dff.copy()
    else:
        yoy_ranking_df = df_calc.copy()

    yoy_ranking_df['year'] = yoy_ranking_df['year_month'].dt.year
    yoy_ranking_df = yoy_ranking_df[yoy_ranking_df['year'].isin([2023, 2024])]

    yoy_agg = (
        yoy_ranking_df.groupby(['brand', 'year'], as_index=False)['search_index']
        .mean()
        .rename(columns={'search_index': 'avg_search_index'})
    )
    yoy_agg = yoy_agg.sort_values(by=['year', 'avg_search_index'], ascending=[True, False])
    yoy_agg['year_rank'] = yoy_agg.groupby('year')['avg_search_index'].rank(method='dense', ascending=False)

    yoy_shifted = yoy_agg.copy()
    yoy_shifted['year'] = yoy_shifted['year'] + 1
    yoy_shifted = yoy_shifted.rename(columns={'year_rank': 'prev_year_rank'})
    merged = pd.merge(
        yoy_agg, yoy_shifted[['brand', 'year', 'prev_year_rank']],
        on=['brand', 'year'], how='left'
    )

    merged['rank_diff'] = merged['prev_year_rank'] - merged['year_rank']
    yoy_table = merged[merged['brand'].isin(selected_brands)].copy()

    rank_pivot = yoy_table.pivot(index='brand', columns='year', values='year_rank')
    diff_pivot = yoy_table.pivot(index='brand', columns='year', values='rank_diff')

    rank_pivot.columns = [f"{c} Rank" for c in rank_pivot.columns]
    diff_pivot.columns = [f"{c} Diff" for c in diff_pivot.columns]

    # Remove "2023 Diff" if it exists
    if "2023 Diff" in diff_pivot.columns:
        diff_pivot.drop(columns=["2023 Diff"], inplace=True, errors='ignore')

    yoy_final = rank_pivot.join(diff_pivot, how='outer').sort_index()

    # Remove any rows where rank/diff are NaN in either 2023 or 2024
    yoy_final.dropna(how='any', inplace=True)

    def color_rank_diff(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: green"
        elif val < 0:
            return "color: red"
        return "color: black"

    diff_cols = [c for c in yoy_final.columns if "Diff" in c]

    if not yoy_final.empty:
        yoy_final_style = (
            yoy_final
            .style
            .format("{:.0f}")
            .applymap(color_rank_diff, subset=diff_cols)
        )
        st.dataframe(
            yoy_final_style,
            use_container_width=False,
            width=1000,
            height=300
        )
    else:
        st.write("No data (2023 or 2024) for the selected brands or missing rank info.")

    # --------------------------------------------------------
    # INDIVIDUAL METRICS
    # --------------------------------------------------------
    for metric in metrics:
        anchor_name = metric.replace("_", "-")
        st.markdown(f'<a name="{anchor_name}"></a>', unsafe_allow_html=True)
        st.subheader(metric)

        dff[metric] = pd.to_numeric(dff[metric], errors='coerce')

        if metric in segment_level_metrics:
            df_metric = dff.copy()
            df_metric.sort_values(by='year_month', inplace=True)
            if df_metric.empty:
                continue
            fig = px.line(
                df_metric,
                x='year_month', y=metric, color='segment',
                title=f"{metric} over time for selected {', '.join(selected_segments)}",
                line_shape='linear'
            )
        else:
            df_metric = dff[dff['brand'].isin(selected_brands)].copy()
            df_metric.sort_values(by='year_month', inplace=True)
            if df_metric.empty:
                continue
            fig = px.line(
                df_metric,
                x='year_month', y=metric, color='brand',
                title=f"{metric} over time for selected {', '.join(selected_segments)}",
                line_shape='linear'
            )

        fig.update_traces(connectgaps=False)
        fig.update_layout(xaxis_title='Date', yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

        desc = metric_descriptions.get(metric, "No description available.")
        st.markdown(f"**Description for {metric}:** {desc}")

    # --------------------------------------------------------
    # ANCHOR: #correlation-heatmap
    # --------------------------------------------------------
    st.markdown('<a name="correlation-heatmap"></a>', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")

    numeric_cols = [
        'global_search_volume', 'segment_total_volume', 'market_share',
        'segment_avg_volume', 'peer_comparison_score', 'qoq_growth',
        'segment_cat_growth_rate', 'geo_diversity', 'seasonality_index',
        'yoy_growth', 'mom_growth', 'contribution_cat_growth', 'search_index'
    ]
    numeric_cols = [c for c in numeric_cols if c in dff.columns]
    if len(numeric_cols) < 2:
        st.write("Not enough numeric metrics to compute correlation.")
    else:
        corr_matrix = dff[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, aspect='auto',
            color_continuous_scale='RdBu', origin='lower'
        )
        fig_corr.update_layout(title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
