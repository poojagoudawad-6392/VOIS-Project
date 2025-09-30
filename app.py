# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
import folium
from folium import plugins
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title='Airbnb Hotel Booking Analysis', layout='wide')

DATA_PATH = os.path.join(os.getcwd(), 'Airbnb_Open_Data(2).zip')

@st.cache_data(show_spinner=False)
def load_data_default(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.rename(columns={'price': 'price_$', 'service fee': 'service_fee_$'}, inplace=True)
    df['price_$'] = df['price_$'].astype(str).replace({'$': '', ',': ''}, regex=True)
    df['service_fee_$'] = df['service_fee_$'].astype(str).replace({'$': '', ',': ''}, regex=True)
    df['price_$'] = pd.to_numeric(df['price_$'], errors='coerce')
    df['service_fee_$'] = pd.to_numeric(df['service_fee_$'], errors='coerce')
    if 'id' in df:
        df['id'] = df['id'].astype(str)
    if 'host id' in df:
        df['host id'] = df['host id'].astype(str)
    if 'last review' in df:
        df['last review'] = pd.to_datetime(df['last review'], errors='coerce')
    if 'Construction year' in df:
        df['Construction year'] = pd.to_numeric(df['Construction year'], errors='coerce')
    if 'neighbourhood group' in df:
        df.loc[df['neighbourhood group'] == 'brookln', 'neighbourhood group'] = 'Brooklyn'
    if drop_outliers:
        if 'availability 365' in df:
            df = df.drop(df[df['availability 365'] > 500].index)
        df = df.drop(df[df['price_$'] > 5000].index)
    if 'service_fee_$' in df:
        df['service_fee_$'] = df['service_fee_$'].fillna(df['service_fee_$'].median())
    if 'reviews per month' in df:
        df['reviews per month'] = df['reviews per month'].fillna(0)
    if 'review rate number' in df:
        df['review rate number'] = df['review rate number'].fillna(df['review rate number'].median())
    if 'minimum nights' in df:
        df['price_per_night'] = df['price_$'] / df['minimum nights']
    df['total_cost'] = df['price_$'] + df['service_fee_$']
    if 'Construction year' in df:
        df['host_experience'] = 2024 - df['Construction year']
    if 'instant_bookable' in df:
        df['is_instant_bookable'] = df['instant_bookable'].map({'TRUE': 1, 'FALSE': 0})
    if 'host_identity_verified' in df:
        df['is_verified'] = df['host_identity_verified'].map({'verified': 1, 'unconfirmed': 0, np.nan: 0})
    # Ensure price_$ has values for modeling/map, with safe imputation if extremely sparse
    if df['price_$'].notna().sum() < 5:
        fallback = (df['service_fee_$'].fillna(0) + 50).clip(lower=10)
        df['price_$'] = df['price_$'].fillna(fallback)
        # If still all NaN, set a constant fallback
        df['price_$'] = df['price_$'].fillna(100)
    return df

@st.cache_data(show_spinner=False)
def model_and_metrics(df: pd.DataFrame):
    work = df.copy()
    if 'price_$' not in work.columns:
        raise ValueError('price_$ column missing after preprocessing.')
    if work['price_$'].notna().sum() < 5:
        median_price = work['price_$'].median()
        if np.isnan(median_price):
            median_price = 100.0
        work['price_$'] = work['price_$'].fillna(median_price)
    work = work.dropna(subset=['price_$']).copy()

    feature_columns_numeric = [
        'minimum nights', 'number of reviews', 'reviews per month',
        'availability 365', 'review rate number', 'service_fee_$',
        'is_instant_bookable', 'is_verified', 'host_experience'
    ]
    existing_numeric = [c for c in feature_columns_numeric if c in work.columns]
    categorical_columns = [c for c in ['neighbourhood group', 'room type'] if c in work.columns]
    work = pd.get_dummies(work, columns=categorical_columns, drop_first=True)
    features = existing_numeric + [c for c in work.columns if c.startswith('neighbourhood group_') or c.startswith('room type_')]
    if len(features) == 0:
        features = existing_numeric
    if len(features) == 0:
        raise ValueError('No usable feature columns found to train model.')
    X = work[features].fillna(0)
    y = work['price_$']

    if len(y) < 5:
        raise ValueError('Not enough rows to train a model.')
    test_size = 0.2 if len(y) >= 10 else max(0.1, min(0.2, len(y) / 5.0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    def metrics(y_true, y_pred):
        return dict(RMSE=math.sqrt(mean_squared_error(y_true, y_pred)), MAE=mean_absolute_error(y_true, y_pred), R2=r2_score(y_true, y_pred))

    m_lr = metrics(y_test, pred_lr)
    m_rf = metrics(y_test, pred_rf)
    try:
        cv = cross_val_score(rf, X, y, cv=min(3, max(2, len(y)//5)), scoring='r2', n_jobs=-1)
    except Exception:
        cv = np.array([np.nan])
    importances = None
    try:
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    except Exception:
        pass
    return (lr, rf, m_lr, m_rf, cv, importances, work, X, features)

@st.cache_data(show_spinner=False)
def build_recommender(df: pd.DataFrame, features: list):
    rec_df = pd.get_dummies(df.copy(), columns=[c for c in ['neighbourhood group', 'room type'] if c in df.columns], drop_first=True)
    X = rec_df[features].fillna(0)
    nn = NearestNeighbors(n_neighbors=6, metric='euclidean').fit(X)
    return nn, rec_df, X

st.title('Airbnb Hotel Booking Analysis')

with st.sidebar:
    st.header('Options')
    keep_all_rows = st.checkbox('Keep all rows (no outlier removal)', value=True)
    use_full_for_models = st.checkbox('Use full dataset for models (ignore filters)', value=True)

if not os.path.exists(DATA_PATH):
    st.error('Airbnb_Open_Data.csv not found in project folder.')
    st.stop()
raw = load_data_default(DATA_PATH)

with st.spinner('Preprocessing data...'):
    df = preprocess(raw, drop_outliers=not keep_all_rows)

with st.sidebar.expander('Data info', expanded=False):
    if 'neighbourhood group' in df.columns:
        st.write('Neighbourhood Groups:', sorted(df['neighbourhood group'].dropna().unique().tolist())[:20])
    if 'room type' in df.columns:
        st.write('Room Types:', sorted(df['room type'].dropna().unique().tolist()))

st.success(f'Data loaded. Records: {len(df):,}')

with st.sidebar:
    st.header('Filters')
    ngroups = sorted([g for g in df.get('neighbourhood group', pd.Series([])).dropna().unique()])
    selected_groups = st.multiselect('Neighbourhood group', ngroups, default=ngroups[:3] if len(ngroups) > 0 else [])
    room_types = sorted([g for g in df.get('room type', pd.Series([])).dropna().unique()])
    selected_rooms = st.multiselect('Room type', room_types, default=room_types[:2] if len(room_types) > 0 else [])

mask = pd.Series(True, index=df.index)
if selected_groups:
    mask &= df['neighbourhood group'].isin(selected_groups)
if selected_rooms:
    mask &= df['room type'].isin(selected_rooms)

view = df[mask]

summary_tab, eda_tab, model_tab, rec_tab, map_tab = st.tabs(['Summary', 'Explore', 'Models', 'Recommendations', 'Map'])

with summary_tab:
    st.subheader('Dataset Overview')
    st.dataframe(view.head(50))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Listings (filtered)', f"{len(view):,}")
    with col2:
        st.metric('Median price', f"${view['price_$'].median():.0f}")
    with col3:
        st.metric('Avg review score', f"{view['review rate number'].mean():.2f}")

with eda_tab:
    st.subheader('Distributions')
    c1, c2 = st.columns(2)
    with c1:
        if 'room type' in view.columns:
            fig = px.histogram(view, x='price_$', color='room type', nbins=50, barmode='overlay', opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'neighbourhood group' in view.columns:
            fig2 = px.box(view, x='neighbourhood group', y='price_$', points='outliers')
            st.plotly_chart(fig2, use_container_width=True)

with model_tab:
    st.subheader('Predictive Models')
    work_len = len(view.dropna(subset=['price_$']))
    use_df = df if use_full_for_models or work_len < 50 else view
    if use_df is df and work_len < 50:
        st.caption('Filtered data too small; models trained on full dataset instead.')
    try:
        lr, rf, m_lr, m_rf, cv, importances, work, X_all, features = model_and_metrics(use_df)
        c1, c2 = st.columns(2)
        with c1:
            st.write('Linear Regression metrics')
            st.json({k: round(v, 3) for k, v in m_lr.items()})
            st.write('Random Forest metrics')
            st.json({k: round(v, 3) for k, v in m_rf.items()})
            if len(cv) > 1 and not np.isnan(cv).all():
                st.write(f"Random Forest CV R2 (3-fold): mean={cv.mean():.3f}, std={cv.std():.3f}")
        with c2:
            if importances is not None:
                st.write('Top feature importances')
                st.bar_chart(importances)
    except ValueError as e:
        st.info(f'Not enough data to train even a minimal model. {e}')

with rec_tab:
    st.subheader('Content-based Recommendations')
    base_for_rec = view if len(view) >= 6 and not use_full_for_models else df
    if len(base_for_rec) >= 6:
        feature_columns_numeric = [
            'minimum nights', 'number of reviews', 'reviews per month',
            'availability 365', 'review rate number', 'service_fee_$',
            'is_instant_bookable', 'is_verified', 'host_experience'
        ]
        existing_numeric = [c for c in feature_columns_numeric if c in base_for_rec.columns]
        cats = [c for c in ['neighbourhood group', 'room type'] if c in base_for_rec.columns]
        tmp = pd.get_dummies(base_for_rec, columns=cats, drop_first=True)
        features = existing_numeric + [c for c in tmp.columns if c.startswith('neighbourhood group_') or c.startswith('room type_')]
        if len(features) == 0:
            features = existing_numeric
        nn, rec_df, recX = build_recommender(base_for_rec, features)
        idx = st.number_input('Listing index (0-based, in chosen dataset)', value=0, min_value=0, max_value=len(rec_df)-1, step=1)
        distances, indices = nn.kneighbors([recX.iloc[idx].values])
        rec_indices = indices[0][1:6]
        cols_to_show = [c for c in ['id', 'NAME', 'neighbourhood group', 'room type', 'price_$', 'review rate number'] if c in rec_df.columns]
        st.write('Similar listings:')
        st.dataframe(rec_df.iloc[rec_indices][cols_to_show])
    else:
        st.info('Not enough listings to generate recommendations.')

with map_tab:
    st.subheader('Geospatial View')
    candidate = view if len(view) >= 10 else df
    def find_latlon(cand: pd.DataFrame):
        if {'latitude', 'longitude'}.issubset(cand.columns):
            return 'latitude', 'longitude'
        if {'lat', 'long'}.issubset(cand.columns):
            return 'lat', 'long'
        return None, None
    lat_col, lon_col = find_latlon(candidate)
    if lat_col is None:
        candidate = df
        lat_col, lon_col = find_latlon(candidate)
    if lat_col and lon_col:
        # Only require lat/lon to plot; do not require price column
        sample_geo = candidate[[lat_col, lon_col]].dropna()
        if len(sample_geo) < 50 and candidate is view:
            candidate = df
            sample_geo = candidate[[lat_col, lon_col]].dropna()
            st.caption('Filtered data had insufficient coordinates; showing map for full dataset.')
        if len(sample_geo) > 0:
            center_lat, center_lon = sample_geo[lat_col].mean(), sample_geo[lon_col].mean()
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            marker_cluster = plugins.MarkerCluster().add_to(fmap)
            for _, row in sample_geo.sample(min(2000, len(sample_geo)), random_state=42).iterrows():
                folium.CircleMarker(location=[row[lat_col], row[lon_col]], radius=3, color='#3388ff', fill=True,
                                    fill_opacity=0.6).add_to(marker_cluster)
            st.components.v1.html(fmap._repr_html_(), height=600, scrolling=False)
        else:
            st.info('No coordinates available.')
    else:
        st.info('Latitude/Longitude columns not found in data.')

st.caption('Built with Streamlit. Use the sidebar to filter results.')

