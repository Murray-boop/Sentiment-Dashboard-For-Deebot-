import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Page Configuration & Helper Functions
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Reddit Listening Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---- Brand palette (match your logo blue) ----
# ---- Brand palette (match your logo blue) ----
BRAND = {
    "deep": "#17374F",
    "mid":  "#21517D",
    "main": "#1A71AE",
    "grid": "rgba(23,55,79,0.12)",
}

def apply_brand_plotly(fig, title=None):
    """
    Make Plotly figures blend with your page "glass card" + brand colors.
    Compatible with more Plotly versions.
    """
    fig.update_layout(
        title=title if title is not None else fig.layout.title.text,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=BRAND["deep"], size=13),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=BRAND["deep"])
        ),
    )

    # Some figures (pie) don't really need axes; still safe to try.
    try:
        fig.update_xaxes(
            showline=False,
            zeroline=False,
            gridcolor=BRAND["grid"],
            tickfont=dict(color=BRAND["deep"]),
            title_font=dict(color=BRAND["deep"]),  # ✅ fixed name
        )
        fig.update_yaxes(
            showline=False,
            zeroline=False,
            gridcolor=BRAND["grid"],
            tickfont=dict(color=BRAND["deep"]),
            title_font=dict(color=BRAND["deep"]),  # ✅ fixed name
        )
    except Exception:
        # For pie-like charts / special cases, ignore axis styling
        pass

    # traces general (safe for bar/line/pie)
    fig.update_traces(
        marker_line_width=0,
        hoverlabel=dict(bgcolor="white", font=dict(color=BRAND["deep"]))
    )

    return fig
THEME_CSS = """
<style>
/* =========================
   0) Hide Streamlit chrome
   ========================= */
header {visibility: hidden;}
[data-testid="stHeader"] {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}

/* =========================
   1) Page background: white -> brand blue gradient
   ========================= */
.stApp {
  background: linear-gradient(180deg,
    #FFFFFF 0%,
    #F3F9FF 28%,
    #E6F2FF 55%,
    #CFE7FF 78%,
    #1A71AE 140%
  );
}

/* Main content container as glass card */
section.main > div.block-container{
  background: rgba(255,255,255,0.82);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: 18px;
  padding: 22px 22px 10px 22px;
  box-shadow: 0 10px 30px rgba(23,55,79,0.15);
  margin-top: 10px;
}

/* Reduce top whitespace */
section.main { padding-top: 0.5rem; }
div.block-container { padding-top: 1.2rem; }

/* =========================
   2) Sidebar: deep blue gradient
   ========================= */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #17374F 0%, #21517D 45%, #1A71AE 110%);
  border-right: 1px solid rgba(255,255,255,0.15);
}

/* Sidebar text (avoid the "sidebar * { color }" trap) */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] small{
  color: rgba(255,255,255,0.92) !important;
}

/* General sidebar controls background (for most inputs) */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input{
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
  color: rgba(255,255,255,0.92) !important;
  -webkit-text-fill-color: rgba(255,255,255,0.92) !important; /* important for Chrome */
}

/* Select dropdown text */
[data-testid="stSidebar"] [data-baseweb="select"] *{
  color: rgba(255,255,255,0.92) !important;
}

/* =========================
   3) DATE RANGE – Force Brand Blue (fix white bg covering white text)
   ========================= */

/* Whole DateInput block becomes a brand panel */
[data-testid="stSidebar"] [data-testid="stDateInput"]{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

/* IMPORTANT: ensure everything calculates border properly */
[data-testid="stSidebar"] [data-testid="stDateInput"] *{
  box-sizing: border-box !important;
}

/* The ONE true frame: apply border/radius here only */
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"]{
  background: linear-gradient(180deg, rgba(33,81,125,0.55) 0%, rgba(26,113,174,0.45) 100%) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;   /* only border */
  border-radius: 16px !important;
  overflow: hidden !important;                            /* clip inner corners */
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.10) !important;

  height: 48px !important;                                /* lock height */
  display: flex !important;
  align-items: center !important;
}

/* Kill any inner wrapper border/background that causes "double frame" */
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="base-input"] > div,
[data-testid="stSidebar"] [data-testid="stDateInput"] div[role="group"],
[data-testid="stSidebar"] [data-testid="stDateInput"] div[role="presentation"]{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  margin: 0 !important;
  padding: 0 !important;
  height: 100% !important;
  display: flex !important;
  align-items: center !important;
}

/* The actual input */
[data-testid="stSidebar"] [data-testid="stDateInput"] input{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;

  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  font-weight: 700 !important;

  height: 100% !important;
  line-height: 48px !important;
  padding: 0 14px !important;
  margin: 0 !important;
  width: 100% !important;
}

/* placeholder */
[data-testid="stSidebar"] [data-testid="stDateInput"] input::placeholder{
  color: rgba(255,255,255,0.60) !important;
  -webkit-text-fill-color: rgba(255,255,255,0.60) !important;
}

/* icon */
[data-testid="stSidebar"] [data-testid="stDateInput"] svg{
  fill: rgba(255,255,255,0.90) !important;
}

/* =========================
   4) MULTISELECT TAG – Force Brand Blue (fix red tags not being hit)
   ========================= */

/* Cover BOTH div and span tag implementations */
[data-testid="stSidebar"] div[data-baseweb="tag"],
[data-testid="stSidebar"] span[data-baseweb="tag"]{
  background: linear-gradient(90deg,#21517D 0%,#1A71AE 100%) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 999px !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.18) !important;
}

/* Tag text inside */
[data-testid="stSidebar"] div[data-baseweb="tag"] span,
[data-testid="stSidebar"] span[data-baseweb="tag"] span{
  color: #FFFFFF !important;
  font-weight: 600 !important;
  -webkit-text-fill-color: #FFFFFF !important;
}

/* Close icon */
[data-testid="stSidebar"] div[data-baseweb="tag"] svg,
[data-testid="stSidebar"] span[data-baseweb="tag"] svg{
  fill: #FFFFFF !important;
}

/* Some versions render the close as a button */
[data-testid="stSidebar"] div[data-baseweb="tag"] button,
[data-testid="stSidebar"] span[data-baseweb="tag"] button{
  color: #FFFFFF !important;
}

/* =========================
   5) Buttons (primary look)
   ========================= */
.stButton>button{
  background: linear-gradient(90deg, #21517D 0%, #1A71AE 100%) !important;
  color: #FFFFFF !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.55rem 1rem !important;
  box-shadow: 0 8px 18px rgba(26,113,174,0.25);
  transition: transform .06s ease, box-shadow .12s ease;
}
.stButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 12px 26px rgba(26,113,174,0.32);
}

/* =========================
   6) Tabs beautify
   ========================= */
button[data-baseweb="tab"]{
  border-radius: 999px !important;
  padding: 8px 14px !important;
  background: rgba(33,81,125,0.08) !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  background: linear-gradient(90deg, rgba(33,81,125,0.18) 0%, rgba(26,113,174,0.22) 100%) !important;
}
div[data-baseweb="tab-list"]{
  gap: 8px;
}

/* =========================
   7) Metric cards look
   ========================= */
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(26,113,174,0.14);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 6px 18px rgba(23,55,79,0.10);
}

/* Titles / text colors in main area */
section.main h1, section.main h2, section.main h3, section.main h4 { color: #17374F; }
section.main p, section.main li, section.main label { color: rgba(23,55,79,0.90); }

/* =========================
   8) File uploader style
   ========================= */
[data-testid="stFileUploader"]{
  background: rgba(255,255,255,0.72);
  border: 1px dashed rgba(26,113,174,0.35);
  border-radius: 16px;
  padding: 10px 12px;
}
[data-testid="stFileUploader"] small{
  color: rgba(23,55,79,0.75) !important;
}

/* =========================
   9) Plotly container blend in
   ========================= */
div[data-testid="stPlotlyChart"]{
  background: rgba(255,255,255,0.58);
  border: 1px solid rgba(26,113,174,0.10);
  border-radius: 16px;
  padding: 10px;
}

/* =========================
   10) Minor: make dividers lighter
   ========================= */
hr{
  border: none;
  border-top: 1px solid rgba(23,55,79,0.12);
}

</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# Helper: Process multi-value columns (mainly for topic and model)
def process_list_col(df, col_name):
    # --- [Core Fix 1]: Restore filling with 'Unknown' to ensure filters capture these rows ---
    df[col_name] = df[col_name].fillna('Unknown')

    # 2. Unify separators: replace Chinese/English commas with semicolons
    df[col_name] = df[col_name].astype(str).str.replace('，', ';').str.replace(',', ';')

    # Replace underscores with spaces (clean data content)
    df[col_name] = df[col_name].str.replace('_', ' ')

    # 3. Split into list, deduplicate, and remove empty items
    df[col_name] = df[col_name].apply(
        lambda x: list(set([item.strip() for item in x.split(';') if item.strip()]))
    )
    return df


# Helper: Get all unique options from a list column
def get_list_unique_options(series):
    all_items = [item for sublist in series for item in sublist]
    return sorted(list(set(all_items)))


# Helper: List filtering logic
def list_filter(row_list, selected_items):
    if not selected_items:  # If nothing selected, default to all (no filter)
        return True
    return not set(row_list).isdisjoint(selected_items)


# --- Model Normalization Function ---
def normalize_models(df):
    if 'model' in df.columns:
        t8_variants = ['T8 AIVI+', 'OZMO T8+', 'T8', 'DEEBOT T8']

        def _clean_single_item(val):
            s = str(val).strip()
            if s in t8_variants:
                return 'T8'
            return s

        def _clean_row_list(row_list):
            if not isinstance(row_list, list):
                return row_list
            cleaned_list = [_clean_single_item(item) for item in row_list]
            # Deduplicate and remove empty values
            cleaned_list = [i for i in cleaned_list if i]
            return list(set(cleaned_list))

        df['model'] = df['model'].apply(_clean_row_list)
    return df


@st.cache_data
def load_data(file):
    df = None
    try:
        # 1. Read file
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='gb18030')
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(file, sheet_name='deep_true')

        # Strip whitespace from columns
        df.columns = df.columns.str.strip()

        # --- Model & Brand Processing ---
        if 'model' not in df.columns:
            if 'sentiment_target' in df.columns:
                df['model'] = df['sentiment_target']
            else:
                df['model'] = None

        # Force Brand
        if 'brand' in df.columns:
            # 取非空、非 Unknown 的值
            brand_series = df['brand'].dropna()
            brand_series = brand_series[brand_series != 'Unknown']

            if not brand_series.empty:
                main_brand = brand_series.value_counts().idxmax()
                df['brand'] = main_brand
            else:
                df['brand'] = 'Unknown'
        else:
            df['brand'] = 'Unknown'

        # --- Date Processing ---
        if 'text_created_utc' in df.columns:
            df['text_created_utc'] = pd.to_datetime(df['text_created_utc'], errors='coerce')

        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].astype(str)
            df = df[df['sentiment'].notna()]
            df = df[df['sentiment'].str.strip() != ""]
            # 如果你也想把字符串 "nan"（由 astype(str) 造成）当成空值删掉：
            df = df[~df['sentiment'].str.strip().str.lower().isin(["nan", "none"])]
        # --- Basic Fill ---
        # [Core Fix 1]: Use 'Unknown' to ensure no data loss
        expected_cols = ['post_subreddit', 'sentiment_target', 'sentiment', 'sentiment_reason']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 'Unknown'
            else:
                df[col] = df[col].fillna('Unknown')

        df['model'] = df['model'].fillna('Unknown')

        # --- Process model column ---
        df = process_list_col(df, 'model')
        df = normalize_models(df)

        # --- Topic Processing ---
        if 'topic' in df.columns:
            df = process_list_col(df, 'topic')
        else:
            df['topic'] = [[] for _ in range(len(df))]

        return df
    except Exception as e:
        st.error(f"Critical error reading file: {e}")
        return None


# -----------------------------------------------------------------------------
# 2. Top Layout
# -----------------------------------------------------------------------------
col_title, col_upload = st.columns([3, 1])
with col_title:
    st.title("📊 Reddit Listening Dashboard")
with col_upload:
    uploaded_file = st.file_uploader("Upload Data File (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

# -----------------------------------------------------------------------------
# 3. Data Processing & Sidebar
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        # --- Basic Data Prep ---
        if 'text_created_utc' not in df.columns:
            st.warning("⚠️ Warning: 'text_created_utc' column not found, date filtering disabled.")

        # Prepare Filter Options ('Unknown' will appear here to allow filtering no-model data)
        all_brands = sorted(df['brand'].astype(str).unique().tolist())
        all_models = get_list_unique_options(df['model'])
        all_sources = sorted(df['post_subreddit'].astype(str).unique().tolist())
        all_topic = get_list_unique_options(df['topic'])
        all_sentiments = sorted(df['sentiment'].astype(str).unique().tolist())

        # ==========================
        # Sidebar: Global Filters
        # ==========================
        st.sidebar.header("🔍 Global Filters")

        # 1. Date Filter
        date_range = None
        if 'text_created_utc' in df.columns and df['text_created_utc'].notnull().any():
            min_date = df['text_created_utc'].min().date()
            max_date = df['text_created_utc'].max().date()
            date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date,
                                               max_value=max_date)

        # 2. Core Dimension Filters
        sb_brands = st.sidebar.multiselect("Brand", all_brands, default=all_brands)
        sb_models = st.sidebar.multiselect("Model", all_models, default=all_models)
        sb_sentiment = st.sidebar.multiselect("Sentiment", all_sentiments, default=all_sentiments)
        sb_sources = st.sidebar.multiselect("Source", all_sources, default=all_sources)
        sb_topic = st.sidebar.multiselect("Topic", all_topic, default=all_topic)

        # Generate Filter Masks
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and 'text_created_utc' in df.columns:
            mask_date = (df['text_created_utc'].dt.date >= date_range[0]) & (
                    df['text_created_utc'].dt.date <= date_range[1])
        else:
            mask_date = True

        mask_brand = df['brand'].isin(sb_brands)
        mask_model = df['model'].apply(lambda x: list_filter(x, sb_models))
        mask_sent = df['sentiment'].isin(sb_sentiment)
        mask_source = df['post_subreddit'].isin(sb_sources)
        mask_topic = df['topic'].apply(lambda x: list_filter(x, sb_topic))

        # Apply Filters
        filtered_df = df[mask_date & mask_brand & mask_model & mask_sent & mask_source & mask_topic]

        # -----------------------------------------------------------------------------
        # 4. Main Interface Tabs
        # -----------------------------------------------------------------------------
        tab_overview, tab_compare = st.tabs(["📈 Overview", "🆚 Comparison Analysis"])

        # ==========================
        # Tab 1: Overview
        # ==========================
        with tab_overview:
            # 1. Core Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Volume", len(filtered_df))

            # Keep '好评' and '差评' in regex in case the input data uses Chinese tags
            pos_count = len(
                filtered_df[filtered_df['sentiment'].str.contains('好评|Positive|positive', case=False, na=False)])
            neg_count = len(
                filtered_df[filtered_df['sentiment'].str.contains('差评|Negative|negative', case=False, na=False)])

            m2.metric("Positive Rate", f"{pos_count / len(filtered_df) * 100:.1f}%" if len(filtered_df) > 0 else "0%")
            m3.metric("Negative Rate", f"{neg_count / len(filtered_df) * 100:.1f}%" if len(filtered_df) > 0 else "0%")
            m4.metric("Models Involved", filtered_df['model'].explode().nunique())

            st.markdown("---")

            # 2. Charts Area
            st.subheader("Volume Trend")
            if not filtered_df.empty and 'text_created_utc' in filtered_df.columns:
                trend_df = filtered_df.groupby(
                    [pd.Grouper(key='text_created_utc', freq='D'), 'sentiment']).size().reset_index(name='count')

                # [Fix] Replace 'Unknown' with empty string in chart
                trend_df['sentiment'] = trend_df['sentiment'].replace('Unknown', '')

                fig_trend = px.bar(
                    trend_df,
                    x='text_created_utc',
                    y='count',
                    color='sentiment',
                    category_orders={"sentiment": ["negative", "neutral", "positive"]},
                    color_discrete_map={'positive': '#636EFA', 'neutral': '#3498db', 'negative': '#e74c3c'},
                    # [Req] Replace underscore in X axis label
                    labels={'text_created_utc': 'text created utc'}
                )
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#17374F")
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No data or date column unavailable")


            st.subheader("Model Sentiment Distribution (Top 20)")
            if not filtered_df.empty:
                df_exploded = filtered_df.explode('model')

                # --- [MODIFIED] Strictly exclude 'Unknown' and empty strings from this chart ---
                df_exploded = df_exploded[~df_exploded['model'].isin(['Unknown', ''])]
                df_exploded = df_exploded[df_exploded['model'].notna()]

                top_models = df_exploded['model'].value_counts().head(20).index

                if len(top_models) > 0:
                    model_sent_df = df_exploded[df_exploded['model'].isin(top_models)].groupby(
                        ['model', 'sentiment']).size().reset_index(name='count')

                    # Clean sentiment column for display
                    model_sent_df['sentiment'] = model_sent_df['sentiment'].replace('Unknown', '')

                    fig_model_sent = px.bar(
                        model_sent_df,
                        x='model', y='count', color='sentiment',
                        category_orders={"sentiment": ["negative", "neutral", "positive"]},
                        color_discrete_map={'positive': '#636EFA', 'neutral': '#3498db', 'negative': '#e74c3c'},
                        barmode='stack'
                    )
                    fig_model_sent = apply_brand_plotly(fig_model_sent, "Model Sentiment Distribution (Top 20)")
                    st.plotly_chart(fig_model_sent, use_container_width=True)
                else:
                    st.info("No valid model data (excluding Unknown)")
            else:
                st.info("No data to display")

            st.subheader("Top 20 Active Users")
            # 尝试匹配常见的用户列名：author 或 author_name
            user_col = None
            for col in ['author', 'author_name', 'user', 'username']:
                if col in filtered_df.columns:
                    user_col = col
                    break

            if user_col and not filtered_df.empty:
                # 统计频次并取 Top 20
                user_counts = filtered_df[user_col].value_counts().head(20).reset_index()
                user_counts.columns = ['User', 'Post Count']

                fig_users = px.bar(
                    user_counts,
                    x='Post Count',
                    y='User',
                    orientation='h',
                    color='Post Count',
                    color_continuous_scale='Blues'
                )
                fig_users.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_users = apply_brand_plotly(fig_users, "Top 20 Active Users")
                st.plotly_chart(fig_users, use_container_width=True)
            else:
                st.info("User column (e.g., 'author') not found in data.")

            st.subheader("Top topic (Top 10)")
            df_exp_topic = filtered_df.explode('topic')
            # [Fix] Replace Unknown for display
            df_exp_topic['topic'] = df_exp_topic['topic'].replace('Unknown', '')

            if not df_exp_topic.empty:
                topic_counts = df_exp_topic['topic'].value_counts().reset_index().head(10)
                topic_counts.columns = ['topic', 'count']

                fig_topic = px.pie(topic_counts, values='count', names='topic', hole=0.4)
                fig_topic = apply_brand_plotly(fig_topic, "Top topic (Top 10)")
                st.plotly_chart(fig_topic, use_container_width=True)
            else:
                st.info("No data to display")

            st.subheader("Source Activity Ranking")
            if not filtered_df.empty:
                display_src = filtered_df.copy()
                display_src['post_subreddit'] = display_src['post_subreddit'].replace('Unknown', '')

                source_counts = display_src['post_subreddit'].value_counts().reset_index().head(10)
                source_counts.columns = ['source', 'count']
                fig_src = px.bar(source_counts, x='count', y='source', orientation='h')
                fig_src.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_src = apply_brand_plotly(fig_src, "Source Activity Ranking")
                st.plotly_chart(fig_src, use_container_width=True)
            else:
                st.info("No data to display")

            # 3. Data Table
            st.subheader("📋 Data Details")
            display_df = filtered_df.copy()


            # [Core Fix 3]: Convert list to string and remove 'Unknown' before display
            # This ensures the table shows empty cells instead of ['Unknown'] or "Unknown"
            def clean_list_for_display(val_list):
                if not isinstance(val_list, list): return str(val_list)
                # Filter out Unknown
                valid_items = [x for x in val_list if x != 'Unknown']
                return ", ".join(valid_items)


            display_df['model'] = display_df['model'].apply(clean_list_for_display)
            display_df['topic'] = display_df['topic'].apply(clean_list_for_display)

            # Clean other normal columns Unknown -> ''
            cols_to_clean = ['brand', 'post_subreddit', 'sentiment', 'sentiment_reason']
            for col in cols_to_clean:
                if col in display_df.columns:
                    display_df[col] = display_df[col].replace('Unknown', '')

            # Define columns to show
            cols_to_show = ['text_created_utc', 'brand', 'model', 'post_subreddit',
                            'sentiment', 'sentiment_reason', 'topic', 'text', 'text_url']
            cols_exist = [c for c in cols_to_show if c in display_df.columns]

            final_display_df = display_df[cols_exist].copy()

            # [Req] Replace underscores with spaces in column names
            new_columns = [c.replace('_', ' ') for c in cols_exist]
            final_display_df.columns = new_columns

            # Sorting
            sort_col = 'text created utc' if 'text created utc' in new_columns else None
            if sort_col:
                final_display_df = final_display_df.sort_values(sort_col, ascending=False)

            st.data_editor(
                final_display_df,
                column_config={"text url": st.column_config.LinkColumn("Link", display_text="Click to Visit")},
                hide_index=True, use_container_width=True, height=400
            )

        # ==========================
        # Tab 2: Comparison Analysis
        # ==========================
        with tab_compare:
            st.markdown("### 🆚 Custom Comparison Analysis")
            st.info("💡 Hint: Compare by **Brand**, **Model**, or **Source**. Leave empty to select all.")


            # --- 定义获取子集数据的函数 ---
            def get_subset_data(raw_df, key_suffix):
                # 布局调整：日期 | 品牌 | 型号 | 来源
                c_date, c_br, c_mo, c_src = st.columns([2, 1, 1, 1])

                d_start, d_end = None, None
                if 'text_created_utc' in raw_df.columns and raw_df['text_created_utc'].notnull().any():
                    d_start = raw_df['text_created_utc'].min()
                    d_end = raw_df['text_created_utc'].max()

                with c_date:
                    if d_start and d_end:
                        dr = st.date_input(f"Date", value=(d_start, d_end), key=f"date_{key_suffix}")
                    else:
                        st.text("Date Unavailable")
                        dr = None

                with c_br:
                    sel_brand = st.multiselect(f"Brand", all_brands, key=f"br_{key_suffix}")
                with c_mo:
                    sel_model = st.multiselect(f"Model", all_models, key=f"mo_{key_suffix}")
                with c_src:
                    sel_src = st.multiselect(f"Source", all_sources, key=f"src_{key_suffix}")

                # 过滤逻辑
                if dr and isinstance(dr, tuple) and len(dr) == 2 and 'text_created_utc' in raw_df.columns:
                    mask_d = (raw_df['text_created_utc'].dt.date >= dr[0]) & (
                            raw_df['text_created_utc'].dt.date <= dr[1])
                else:
                    mask_d = True

                mask_b = raw_df['brand'].isin(sel_brand) if sel_brand else True

                # --- [核心修复]：这里改为使用 list_filter ---
                mask_m = raw_df['model'].apply(lambda x: list_filter(x, sel_model))

                mask_s = raw_df['post_subreddit'].isin(sel_src) if sel_src else True

                return raw_df[mask_d & mask_b & mask_m & mask_s]


            with st.container():
                st.markdown("#### 🅰️ Group A")
                df_a = get_subset_data(df, "A")
            st.divider()
            with st.container():
                st.markdown("#### 🅱️ Group B")
                df_b = get_subset_data(df, "B")
            st.divider()

            if len(df_a) == 0 and len(df_b) == 0:
                st.warning("No data.")
            else:
                st.markdown("#### 📊 Comparison Results")

                kpi1, kpi2, kpi3 = st.columns(3)
                vol_a, vol_b = len(df_a), len(df_b)
                kpi1.metric("Total Volume (B vs A)", f"{vol_b}", delta=f"{vol_b - vol_a}")


                def calc_rate(d, keyword):
                    if len(d) == 0: return 0.0
                    cnt = len(d[d['sentiment'].str.contains(keyword, case=False, na=False)])
                    return (cnt / len(d)) * 100


                pos_a, pos_b = calc_rate(df_a, '好评|Positive|positive'), calc_rate(df_b, '好评|Positive|positive')
                neg_a, neg_b = calc_rate(df_a, '差评|Negative|negative'), calc_rate(df_b, '差评|Negative|negative')

                kpi2.metric("Positive Rate", f"{pos_b:.1f}%", delta=f"{pos_b - pos_a:.1f}%")
                kpi3.metric("Negative Rate", f"{neg_b:.1f}%", delta=f"{neg_b - neg_a:.1f}%", delta_color="inverse")


                st.markdown("**Top Topic Comparison**")


                def get_top_topic(d, group_name):
                    if len(d) == 0: return pd.DataFrame()
                    d_exp = d.explode('topic')
                    # Replace Unknown in comparison chart too
                    d_exp['topic'] = d_exp['topic'].replace('Unknown', '')
                    res = d_exp['topic'].value_counts().head(5).reset_index()
                    res.columns = ['topic', 'count']
                    res['Group'] = group_name
                    return res


                top_a = get_top_topic(df_a, 'Group A')
                top_b = get_top_topic(df_b, 'Group B')
                comb_topic = pd.concat([top_a, top_b])

                if not comb_topic.empty:
                    fig_topic_cmp = px.bar(comb_topic, x='topic', y='count', color='Group', barmode='group')
                    fig_topic_cmp = apply_brand_plotly(fig_topic_cmp, "Top Topic Comparison")
                    st.plotly_chart(fig_topic_cmp, use_container_width=True)
                else:
                    st.info("No data")

                st.markdown("**Model Distribution Comparison**")
                if 'model' in df.columns:
                    df_a_exp = df_a.explode('model')
                    df_b_exp = df_b.explode('model')

                    # [Fix] Replace Unknown
                    df_a_exp['model'] = df_a_exp['model'].replace('Unknown', '')
                    df_b_exp['model'] = df_b_exp['model'].replace('Unknown', '')

                    df_a_m = df_a_exp[['model']].copy()
                    df_a_m['Group'] = 'Group A'
                    df_b_m = df_b_exp[['model']].copy()
                    df_b_m['Group'] = 'Group B'

                    merged_mod = pd.concat([df_a_m, df_b_m])
                    merged_mod = merged_mod.dropna(subset=['model'])
                    # Remove empty string models from comparison chart as well
                    merged_mod = merged_mod[merged_mod['model'] != '']

                    if not merged_mod.empty:
                        comp_mod = merged_mod.groupby(['Group', 'model']).size().reset_index(name='count')
                        comp_mod['percentage'] = comp_mod.groupby('Group')['count'].transform(lambda x: x / x.sum())
                        fig_model_cmp = px.bar(
                            comp_mod, x='Group', y='percentage', color='model',
                            title="Model Share (100% Stacked)", text_auto='.1%'
                        )
                        fig_model_cmp = apply_brand_plotly(fig_model_cmp, "Model Distribution Comparison")
                        st.plotly_chart(fig_model_cmp, use_container_width=True)
                    else:
                        st.info("No data")
                else:
                    st.info("No model data")
else:
    st.info("👋 Please upload a data file (must include brand, model, sentiment_reason, etc.) in the top right corner.")

