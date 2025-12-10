import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Page Configuration & Helper Functions
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Reddit Listening Dashboard for Deebot",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
            df = pd.read_excel(file)

        # Strip whitespace from columns
        df.columns = df.columns.str.strip()

        # --- Model & Brand Processing ---
        if 'model' not in df.columns:
            if 'sentiment_target' in df.columns:
                df['model'] = df['sentiment_target']
            else:
                df['model'] = None

        # Force Brand
        df['brand'] = 'DEEBOT'

        # --- Date Processing ---
        if 'text_created_utc' in df.columns:
            df['text_created_utc'] = pd.to_datetime(df['text_created_utc'], errors='coerce')

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
    st.title("📊 Reddit Listening Dashboard for Deebot")
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
        all_topics = get_list_unique_options(df['topic'])
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
        sb_topics = st.sidebar.multiselect("Topic", all_topics, default=all_topics)

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
        mask_topic = df['topic'].apply(lambda x: list_filter(x, sb_topics))

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
            m1.metric("Total Volume (Posts)", len(filtered_df))

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
            c1, c2 = st.columns(2)
            with c1:
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
                        color_discrete_map={'positive': '#e74c3c', 'neutral': '#3498db', 'negative': '#636EFA'},
                        # [Req] Replace underscore in X axis label
                        labels={'text_created_utc': 'text created utc'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No data or date column unavailable")

            with c2:
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

                        st.plotly_chart(
                            px.bar(model_sent_df, x='model', y='count', color='sentiment',
                                   color_discrete_map={'positive': '#e74c3c', 'neutral': '#3498db',
                                                       'negative': '#636EFA'},
                                   barmode='stack'),
                            use_container_width=True)
                    else:
                        st.info("No valid model data (excluding Unknown)")
                else:
                    st.info("No data to display")

            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Top Topics (Top 10)")
                df_exp_topic = filtered_df.explode('topic')
                # [Fix] Replace Unknown for display
                df_exp_topic['topic'] = df_exp_topic['topic'].replace('Unknown', '')

                if not df_exp_topic.empty:
                    topic_counts = df_exp_topic['topic'].value_counts().reset_index().head(10)
                    topic_counts.columns = ['topic', 'count']

                    st.plotly_chart(px.pie(topic_counts, values='count', names='topic', hole=0.4),
                                    use_container_width=True)
                else:
                    st.info("No data to display")

            with c4:
                st.subheader("Source Activity Ranking")
                if not filtered_df.empty:
                    display_src = filtered_df.copy()
                    display_src['post_subreddit'] = display_src['post_subreddit'].replace('Unknown', '')

                    source_counts = display_src['post_subreddit'].value_counts().reset_index().head(10)
                    source_counts.columns = ['source', 'count']
                    fig_src = px.bar(source_counts, x='count', y='source', orientation='h')
                    fig_src.update_layout(yaxis={'categoryorder': 'total ascending'})
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

                chart1, chart2 = st.columns(2)

                with chart1:
                    st.markdown("**Top Topic Comparison**")


                    def get_top_topics(d, group_name):
                        if len(d) == 0: return pd.DataFrame()
                        d_exp = d.explode('topic')
                        # Replace Unknown in comparison chart too
                        d_exp['topic'] = d_exp['topic'].replace('Unknown', '')
                        res = d_exp['topic'].value_counts().head(5).reset_index()
                        res.columns = ['topic', 'count']
                        res['Group'] = group_name
                        return res


                    top_a = get_top_topics(df_a, 'Group A')
                    top_b = get_top_topics(df_b, 'Group B')
                    comb_topic = pd.concat([top_a, top_b])

                    if not comb_topic.empty:
                        st.plotly_chart(px.bar(comb_topic, x='topic', y='count', color='Group', barmode='group'),
                                        use_container_width=True)
                    else:
                        st.info("No data")

                with chart2:
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
                            st.plotly_chart(px.bar(comp_mod, x='Group', y='percentage', color='model',
                                                   title="Model Share (100% Stacked)", text_auto='.1%'),
                                            use_container_width=True)
                        else:
                            st.info("No data")
                    else:
                        st.info("No model data")
else:
    st.info("👋 Please upload a data file (must include brand, model, sentiment_reason, etc.) in the top right corner.")