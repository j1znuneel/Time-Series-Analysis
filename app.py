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
                outlier_method = st.selectbox("Outlier Detection Method", ["Z-Score", "IQR", "Hampel Filter", "Isolation Forest", "DBSCAN"], key=f"method_{station_name}")
            
            # 3.5 SENSITIVITY CONTROLS
            sens_col1, sens_col2 = st.columns([1, 3])
            with sens_col1:
                st.write("")
                st.write("**Detection Sensitivity:**")
            with sens_col2:
                if outlier_method == "Z-Score":
                    threshold = st.slider("Z-Score Threshold", 2.0, 10.0, 3.5, 0.5, key=f"thresh_{station_name}")
                elif outlier_method == "IQR":
                    threshold = st.slider("IQR Multiplier", 1.0, 5.0, 1.5, 0.5, key=f"thresh_{station_name}")
                elif outlier_method == "Hampel Filter":
                    threshold = st.slider("Sigma Threshold", 2.0, 10.0, 3.0, 0.5, key=f"thresh_{station_name}")
                else:
                    threshold = 3.0
            
            # 4. TRENDING / DECOMPOSITION
            data_raw_mm = (work_df[col_select] - work_df[col_select].mean()) * 1000
            wrms_before = calculate_wrms(data_raw_mm, work_df[sig_col] * 1000)
            
            if model_type == "Linear Trend":
                z = np.polyfit(work_df.index, data_raw_mm, 1)
                p = np.poly1d(z)
                work_df['trend_mm'] = p(work_df.index)
                work_df['seasonal_mm'] = 0
                work_df['residual_mm'] = data_raw_mm - work_df['trend_mm']
                model_label = "Linear Trend"
            else:
                mjd_range = np.arange(work_df['mjd'].min(), work_df['mjd'].max() + 1)
                interp_df = work_df.reset_index().set_index('mjd')
                interp_df = interp_df.reindex(mjd_range)
                data_mean = work_df[col_select].mean()
                interp_data_mm = (interp_df[col_select] - data_mean) * 1000
                interp_data_mm = interp_data_mm.interpolate(method='linear')
                
                try:
                    period = 365
                    if len(interp_data_mm) < period * 2:
                        period = max(3, (len(interp_data_mm) // 2) // 2 * 2 + 1)
                    
                    stl = STL(interp_data_mm, period=period, robust=True)
                    res = stl.fit()
                    
                    work_df['trend_mm'] = res.trend.reindex(work_df['mjd'].values).values
                    work_df['seasonal_mm'] = res.seasonal.reindex(work_df['mjd'].values).values
                    work_df['residual_mm'] = res.resid.reindex(work_df['mjd'].values).values
                except Exception as e:
                    st.warning(f"STL failed: {e}")
                    z = np.polyfit(work_df.index, data_raw_mm, 1)
                    p = np.poly1d(z)
                    work_df['trend_mm'] = p(work_df.index)
                    work_df['seasonal_mm'] = 0
                    work_df['residual_mm'] = data_raw_mm - work_df['trend_mm']
                model_label = "Trend+Seasonal"

            # 5. OUTLIER DETECTION (Variables are now defined)
            target_data = work_df['residual_mm'].fillna(0)
            
            if outlier_method == "Z-Score":
                outliers = get_outliers_zscore(target_data, threshold=threshold)
            elif outlier_method == "IQR":
                outliers = get_outliers_iqr(target_data, k=threshold)
            elif outlier_method == "Hampel Filter":
                outliers = get_outliers_hampel(target_data, n_sigmas=threshold)
            elif outlier_method == "Isolation Forest":
                outliers = get_outliers_iso_forest(target_data)
            else:
                outliers = get_outliers_dbscan(target_data)
                
            work_df['is_outlier'] = outliers
            num_outliers = outliers.sum()
            
            clean_df = work_df[~work_df['is_outlier']]
            wrms_after = calculate_wrms(clean_df[col_select] * 1000 - (work_df[col_select].mean()*1000), 
                                       clean_df[sig_col] * 1000, 
                                       model_fit=(clean_df['trend_mm'] + clean_df['seasonal_mm']))
            
            # Plot
            st.subheader(f"🔍 Detailed Outlier Analysis: {comp_base.capitalize()} (mm)")
            fig_detail = go.Figure()
            fig_detail.add_trace(go.Scatter(x=work_df.index, y=data_raw_mm, mode='markers', name='Data', marker=dict(size=4, color='gray', opacity=0.5)))
            fig_detail.add_trace(go.Scatter(x=work_df.index, y=work_df['trend_mm'] + work_df['seasonal_mm'], mode='lines', name=model_label, line=dict(color='yellow')))
            
            outlier_df = work_df[work_df['is_outlier']]
            fig_detail.add_trace(go.Scatter(x=outlier_df.index, y=data_raw_mm[work_df['is_outlier']], mode='markers', name='Outliers', marker=dict(size=7, color='red', symbol='x')))
            
            fig_detail.update_layout(xaxis_title="Decimal Year", yaxis_title="Displacement (mm)")
            fig_detail.update_xaxes(tickformat=".4f")
            st.plotly_chart(fig_detail, width='stretch')
            
            results_list.append({
                "Station": station_name,
                "Method": outlier_method,
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
