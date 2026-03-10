import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import datetime
from plotly.subplots import make_subplots

st.set_page_config(page_title="GNSS Time Series Analysis", layout="wide")

def parse_tenv3(file):
    try:
        # Standard 23-column format
        cols = ['site', 'yymmdd', 'dec_year', 'mjd', 'week', 'd', 'reflon', 'e0', 'east', 
                'n0', 'north', 'u0', 'up', 'ant', 'sig_e', 'sig_n', 'sig_u', 
                'c_en', 'c_eu', 'c_nu', 'lat', 'lon', 'height']
        
        df = pd.read_csv(file, sep=r'\s+', names=cols, skiprows=1, comment='#')
        
        # Ensure numeric
        numeric_cols = ['dec_year', 'mjd', 'e0', 'east', 'n0', 'north', 'u0', 'up', 'sig_e', 'sig_n', 'sig_u']
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Calculate Totals
        df['east_total'] = df['e0'] + df['east']
        df['north_total'] = df['n0'] + df['north']
        df['up_total'] = df['u0'] + df['up']
        
        # Convert to datetime for UI display
        def parse_date(d):
            try:
                return datetime.datetime.strptime(str(d), "%y%b%d")
            except:
                return pd.NaT
        
        df['date'] = df['yymmdd'].apply(parse_date)
        df = df.dropna(subset=['mjd', 'east_total'])
        df = df.sort_values('mjd')
        return df
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def find_longest_span(df):
    if df.empty:
        return None, None, 0, None
    
    df = df.sort_values('mjd')
    df['gap'] = df['mjd'].diff()
    
    # Create groups where a gap > 7 days starts a new group
    df['group_id'] = (df['gap'] > 7).cumsum()
    
    # Find the group with the most rows
    counts = df['group_id'].value_counts()
    longest_group_id = counts.idxmax()
    
    df_best = df[df['group_id'] == longest_group_id].copy()
    
    start_val = df_best['dec_year'].min()
    end_val = df_best['dec_year'].max()
    span_points = len(df_best)
    
    return start_val, end_val, span_points, df_best

def get_outliers_zscore(data, threshold=3.5):
    z = np.abs(stats.zscore(data, nan_policy='omit'))
    return z > threshold

def get_outliers_iqr(data, k=1.5):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    return (data < (q1 - k * iqr)) | (data > (q3 + k * iqr))

def get_outliers_hampel(data, window_size=10, n_sigmas=3.0):
    rolling_median = data.rolling(window=window_size, center=True).median()
    rolling_mad = data.rolling(window=window_size, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    upper_bound = rolling_median + n_sigmas * 1.4826 * rolling_mad
    lower_bound = rolling_median - n_sigmas * 1.4826 * rolling_mad
    return (data > upper_bound) | (data < lower_bound)

def get_outliers_iso_forest(data, contamination=0.01):
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(data.values.reshape(-1, 1))
    return preds == -1

def get_outliers_dbscan(data, eps=0.5, min_samples=5):
    scaled = (data - data.mean()) / data.std()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    preds = model.fit_predict(scaled.values.reshape(-1, 1))
    return preds == -1

def calculate_wrms(data, sigma, model_fit=None):
    if model_fit is None:
        model_fit = np.nanmean(data)
    
    residuals = data - model_fit
    weights = 1.0 / (sigma**2)
    mask = ~np.isnan(residuals) & ~np.isnan(weights)
    
    if not np.any(mask):
        return 0
        
    res = residuals[mask]
    w = weights[mask]
    wrms = np.sqrt(np.sum(w * (res**2)) / np.sum(w))
    return wrms

st.title("🛰️ GNSS Time Series Analysis")
st.markdown("""
Upload your `.tenv3` files to analyze GNSS position time series.
""")

uploaded_files = st.file_uploader("Upload .tenv3 files", type=["tenv3"], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    
    for uploaded_file in uploaded_files:
        st.divider()
        df = parse_tenv3(uploaded_file)
        
        if df is not None:
            station_name = df['site'].iloc[0] if not df['site'].empty else uploaded_file.name
            st.header(f"📍 Station: {station_name}")
            
            # 1. Longest Span Detection
            start_year, end_year, span_points, work_df = find_longest_span(df)
            st.info(f"📅 **Best Stretch:** {start_year:.4f} to {end_year:.4f} ({span_points} data points)")
            
            work_df = work_df.set_index('dec_year')
            
            # 2. OVERVIEW VISUALIZATION
            st.subheader("🔭 Displacement Overview (Millimeters)")
            
            fig_ov = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                 vertical_spacing=0.05,
                                 subplot_titles=("East (mm)", "North (mm)", "Up (mm)"))
            
            for i, (comp, color) in enumerate(zip(['east_total', 'north_total', 'up_total'], ['blue', 'green', 'red']), 1):
                data_centered = (work_df[comp] - work_df[comp].mean()) * 1000
                fig_ov.add_trace(
                    go.Scatter(x=work_df.index, y=data_centered, 
                               mode='markers', name=comp.replace('_total','').capitalize(),
                               marker=dict(size=3, color=color)),
                    row=i, col=1
                )
            
            fig_ov.update_layout(height=800, showlegend=False, hovermode="x unified")
            fig_ov.update_xaxes(tickformat=".4f", row=3, col=1, title="Decimal Year")
            st.plotly_chart(fig_ov, width='stretch')

            # 3. ANALYSIS CONTROLS
            col1, col2, col3 = st.columns(3)
            with col1:
                comp_base = st.selectbox(f"Select Component", ["east", "north", "up"], key=f"comp_{station_name}")
                col_select = f"{comp_base}_total"
                sig_col = f"sig_{comp_base[0]}"
            with col2:
                model_type = st.selectbox("Trend Model Type", ["Linear Trend", "Seasonal Decomposition (STL)"], key=f"model_{station_name}")
            with col3:
                model_smoothness = st.slider("Model Smoothness (higher = stiffer line)", 5, 151, 35, 10, key=f"smooth_{station_name}")
            
            # 3.2 SENSITIVITY & OUTLIER METHOD
            out_col1, out_col2, out_col3 = st.columns([1, 1, 2])
            with out_col1:
                fill_method = st.selectbox("Gap Filling", ["None", "Linear Interpolation", "Forward Fill"], index=1, key=f"fill_{station_name}")
            with out_col2:
                outlier_method = st.selectbox("Detection Method", ["Z-Score", "IQR", "Hampel Filter", "Isolation Forest"], key=f"method_{station_name}")
            with out_col3:
                if outlier_method == "Z-Score":
                    threshold = st.slider("Z-Threshold", 1.5, 7.0, 3.0, 0.5, key=f"thresh_{station_name}")
                elif outlier_method == "IQR":
                    threshold = st.slider("IQR Multiplier", 0.5, 4.0, 1.2, 0.1, key=f"thresh_{station_name}")
                elif outlier_method == "Hampel Filter":
                    threshold = st.slider("Sigma", 1.5, 6.0, 3.0, 0.5, key=f"thresh_{station_name}")
                else:
                    threshold = 3.0

            # 4. GAP FILLING
            full_mjd = np.arange(work_df['mjd'].min(), work_df['mjd'].max() + 1)
            work_df = work_df.reset_index().set_index('mjd').reindex(full_mjd)
            work_df['is_interpolated'] = work_df[col_select].isna()
            
            # Explicitly convert numeric columns before interpolation
            numeric_cols_fix = [col_select, sig_col, 'dec_year']
            for c in numeric_cols_fix:
                if c in work_df.columns:
                    work_df[c] = pd.to_numeric(work_df[c], errors='coerce')
            
            if fill_method == "Linear Interpolation":
                work_df = work_df.interpolate(method='linear')
            elif fill_method == "Forward Fill":
                work_df = work_df.ffill()
            
            if work_df['dec_year'].isna().any():
                work_df['dec_year'] = work_df['dec_year'].interpolate(method='linear')
            
            # 5. ADVANCED TWO-STAGE OUTLIER DETECTION
            data_mean = work_df[col_select].mean()
            data_raw_mm = (work_df[col_select] - data_mean) * 1000
            obs_mask = ~work_df['is_interpolated']
            
            # STAGE 1: ROLLING MEDIAN BASELINE (immune to spikes)
            detection_baseline = data_raw_mm.rolling(window=91, center=True, min_periods=1).median()
            detection_baseline = detection_baseline.ffill().bfill()
            residuals_for_detection = data_raw_mm - detection_baseline
            
            # STAGE 2: Run Detection
            if outlier_method == "Z-Score":
                outliers = get_outliers_zscore(residuals_for_detection, threshold=threshold)
            elif outlier_method == "IQR":
                outliers = get_outliers_iqr(residuals_for_detection, k=threshold)
            elif outlier_method == "Hampel Filter":
                outliers = get_outliers_hampel(data_raw_mm, n_sigmas=threshold, window_size=26)
            else:
                outliers = get_outliers_iso_forest(residuals_for_detection)
                
            work_df['is_outlier'] = outliers
            num_outliers = outliers.sum()
            
            # STAGE 3: Final Analysis on CLEANED data
            clean_df = work_df[~work_df['is_outlier']].copy()
            clean_full_mm = data_raw_mm.copy()
            clean_full_mm[work_df['is_outlier']] = np.nan
            clean_full_mm = clean_full_mm.interpolate(method='linear').ffill().bfill()
            
            if model_type == "Linear Trend":
                z_final = np.polyfit(work_df['dec_year'], clean_full_mm, 1)
                p_final = np.poly1d(z_final)
                work_df['trend_mm'] = p_final(work_df['dec_year'])
                work_df['seasonal_mm'] = 0
                model_label = "Linear Trend"
            else:
                try:
                    period = 365
                    if len(clean_full_mm) < period * 2:
                        period = max(3, (len(clean_full_mm) // 2) // 2 * 2 + 1)
                    
                    sw = model_smoothness if model_smoothness % 2 != 0 else model_smoothness + 1
                    stl = STL(clean_full_mm, period=period, seasonal=sw, robust=True)
                    res = stl.fit()
                    work_df['trend_mm'] = res.trend
                    work_df['seasonal_mm'] = res.seasonal
                except Exception as e:
                    st.warning(f"STL failed: {e}")
                    z_f = np.polyfit(work_df['dec_year'], clean_full_mm, 1)
                    p_f = np.poly1d(z_f)
                    work_df['trend_mm'] = p_f(work_df['dec_year'])
                    work_df['seasonal_mm'] = 0
                model_label = "Trend+Seasonal"

            # Final Residuals for WRMS Calculation
            work_df['residual_mm'] = data_raw_mm - (work_df['trend_mm'] + work_df['seasonal_mm'])
            wrms_before = calculate_wrms(data_raw_mm[obs_mask], work_df.loc[obs_mask, sig_col] * 1000)
            
            clean_mask = (~work_df['is_outlier']) & obs_mask
            wrms_after = calculate_wrms(work_df.loc[clean_mask, 'residual_mm'], 
                                       work_df.loc[clean_mask, sig_col] * 1000)
            
            # Velocity Calculation
            velocity_mmyr = np.polyfit(work_df.loc[~work_df['is_outlier'], 'dec_year'], 
                                       clean_full_mm[~work_df['is_outlier']], 1)[0]
            
            st.success(f"🚀 **Estimated Plate Velocity ({comp_base.capitalize()}):** {velocity_mmyr:.3f} mm/yr")
            
            # Plot
            st.subheader(f"🔍 Detailed Outlier Analysis: {comp_base.capitalize()} (mm)")
            fig_detail = go.Figure()
            fig_detail.add_trace(go.Scatter(x=work_df['dec_year'], y=data_raw_mm, mode='markers', name='Data', marker=dict(size=4, color='gray', opacity=0.5)))
            fig_detail.add_trace(go.Scatter(x=work_df['dec_year'], y=work_df['trend_mm'] + work_df['seasonal_mm'], mode='lines', name=model_label, line=dict(color='yellow')))
            
            outlier_df = work_df[work_df['is_outlier']]
            fig_detail.add_trace(go.Scatter(x=outlier_df['dec_year'], y=data_raw_mm[work_df['is_outlier']], mode='markers', name='Outliers', marker=dict(size=7, color='red', symbol='x')))
            
            fig_detail.update_layout(xaxis_title="Decimal Year", yaxis_title="Displacement (mm)")
            fig_detail.update_xaxes(tickformat=".4f")
            st.plotly_chart(fig_detail, width='stretch')
            
            results_list.append({
                "Station": station_name,
                "Component": comp_base.capitalize(),
                "Velocity (mm/yr)": round(velocity_mmyr, 3),
                "Outliers Detected": num_outliers,
                "WRMS Before (mm)": round(wrms_before, 3),
                "WRMS After (mm)": round(wrms_after, 3)
            })

    if results_list:
        st.header("📋 Final Results Summary")
        results_df = pd.DataFrame(results_list)
        st.table(results_df)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, "tsa_results.csv", "text/csv", key='download-csv')
else:
    st.info("Please upload one or more .tenv3 files to begin.")
