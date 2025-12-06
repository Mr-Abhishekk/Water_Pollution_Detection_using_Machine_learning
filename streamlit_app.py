import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Water Pollution Prediction", layout="wide")

# --- 2. Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #1E1E26 0%, #0F0F12 100%);
        color: #FFFFFF;
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #1E1E26;
        border-right: 1px solid #333333;
    }
    
    [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: #FFFFFF;
    }

    /* Tab buttons style */
    [data-testid="stTabs"] button {
        background-color: transparent;
        color: #AAAAAA;
        border: 1px solid #333333;
        border-radius: 8px;
        margin-right: 10px;
        transition: all 0.3s;
    }
    
    [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #0068C9;
        color: white;
        border: 1px solid #0068C9;
    }
    
    [data-testid="stTabs"] button:hover {
        background-color: #2F2F3D;
        color: white;
    }

    /* Prediction result boxes */
    [data-testid="stError"] {
        background-color: #4C1B1B;
        border: 1px solid #FF4B4B;
        border-radius: 8px;
    }
    [data-testid="stWarning"] {
        background-color: #4B381A;
        border: 1px solid #FFC04B;
        border-radius: 8px;
    }
    [data-testid="stSuccess"] {
        background-color: #1A4B2A;
        border: 1px solid #4BFF8D;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)


st.title("💧 Water Pollution Detection App")

# --- 3. Define Full File Paths ---
MODEL_PATH = r"C:\Users\VICTUS\Documents\skills4future\Water_Pollution_Detection\model\dt_model_pipeline.joblib"
ENCODER_PATH = r"C:\Users\VICTUS\Documents\skills4future\Water_Pollution_Detection\model\label_encoder (1).joblib"
CSV_PATH = r"C:\Users\VICTUS\Documents\skills4future\Water_Pollution_Detection\data\water_dataX.csv"

# --- 4. Load the Saved Model and Encoder ---
try:
    pipeline = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("Model and encoder loaded successfully.")
except FileNotFoundError:
    st.error(f"ERROR: Model or encoder files not found at the specified paths.")
    st.stop() 

# --- 5. Define Feature Names and Limits ---
features = ['do', 'ph', 'conductivity', 'bod', 'nitrate', 'temp', 'fecal_coliform', 'total_coliform']
feature_names = {
    'do': 'Dissolved Oxygen', 'ph': 'pH', 'conductivity': 'Conductivity',
    'bod': 'Biochemical Oxygen Demand', 'nitrate': 'Nitrate + Nitrite', 'temp': 'Temperature',
    'fecal_coliform': 'Fecal Coliform', 'total_coliform': 'Total Coliform'
}
safe_limits = {
    'do': {'Ideal': '≥ 6', 'Polluted': '< 4'},
    'ph': {'Ideal': '6.5 – 8.5', 'Polluted': '< 6.0 or > 9.0'},
    'bod': {'Ideal': '0 – 3', 'Polluted': '> 6'},
    'conductivity': {'Ideal': '200 – 500', 'Polluted': '> 1000'},
    'nitrate': {'Ideal': '0 – 10', 'Polluted': '> 45'},
    'temp': {'Ideal': '20 – 30 °C', 'Polluted': '> 35 °C'},
    'fecal_coliform': {'Ideal': '0 – 10', 'Polluted': '> 100'},
    'total_coliform': {'Ideal': '0 – 50', 'Polluted': '> 500'}
}

# --- 6. Data Loading and Cleaning Function ---
@st.cache_data
def load_and_clean_data(filepath):
    """Loads and cleans the uploaded file."""
    try:
        df = pd.read_csv(filepath, encoding='iso-8859-1')
    except Exception:
        df = pd.read_csv(filepath, encoding='latin1')

    # Rename all columns
    df.rename(columns={
        'D.O. (mg/l)': 'do', 'PH': 'ph', 'CONDUCTIVITY (µmhos/cm)': 'conductivity',
        'B.O.D. (mg/l)': 'bod', 'NITRATENAN N+ NITRITENANN (mg/l)': 'nitrate',
        'FECAL COLIFORM (MPN/100ml)': 'fecal_coliform',
        'TOTAL COLIFORM (MPN/100ml)Mean': 'total_coliform', 'Temp': 'temp'
    }, inplace=True)

    # Clean feature columns
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean non-feature columns
    df['STATE'] = df['STATE'].fillna('Unknown').str.strip().str.title()
    df['LOCATIONS'] = df['LOCATIONS'].fillna('Unknown').str.strip().str.title()
    df['STATION CODE'] = df['STATION CODE'].fillna('Unknown')
    df['year'] = df['year'].fillna(0).astype(int)

    # Clean outliers
    df['ph'] = df['ph'].apply(lambda x: x if (pd.isna(x) or (0 <= x <= 14)) else np.nan)
    for col in ['conductivity', 'fecal_coliform', 'total_coliform', 'bod', 'nitrate']:
        if col in df.columns:
            q = df[col].quantile(0.999)
            if pd.notna(q):
                df[col] = df[col].apply(lambda x: q if x > q else x)
    
    return df

# --- 7. Load Data Automatically ---
try:
    df_clean = load_and_clean_data(CSV_PATH)
    data_ranges = df_clean[features].describe().loc[['min', 'max']]
except FileNotFoundError:
    st.error(f"ERROR: '{CSV_PATH}' not found. App cannot load.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the data: {e}")
    st.stop()

# --- 8. Create Tabs ---
tab1, tab2 = st.tabs(["Predict Manually", "Search by Location"])

# --- TAB 1: PREDICT MANUALLY (Using Number Inputs) ---
with tab1:
    with st.form("manual_prediction_form"):
        st.header("Enter Water Quality Values")
        
        c1, c2 = st.columns(2)
        input_data = {}
        
        with c1:
            st.subheader("Physical & Chemical")
            for feature in features[:4]:
                full_name = feature_names[feature]
                min_val = float(data_ranges.loc['min', feature])
                max_val = float(data_ranges.loc['max', feature])
                median_val = float(df_clean[feature].median())
                input_data[feature] = st.number_input(
                    label=f"{full_name} ({feature})",
                    min_value=min_val, max_value=max_val,
                    value=median_val, step=0.1, key=f"manual_{feature}"
                )
        with c2:
            st.subheader("Biological & Other")
            for feature in features[4:]:
                full_name = feature_names[feature]
                min_val = float(data_ranges.loc['min', feature])
                max_val = float(data_ranges.loc['max', feature])
                median_val = float(df_clean[feature].median())
                input_data[feature] = st.number_input(
                    label=f"{full_name} ({feature})",
                    min_value=min_val, max_value=max_val,
                    value=median_val, step=0.1, key=f"manual_{feature}"
                )
        submit_button = st.form_submit_button(label='Predict Water Quality')

    if submit_button:
        input_df = pd.DataFrame([input_data])[features]
        prediction_numeric = pipeline.predict(input_df)
        prediction_label = encoder.inverse_transform(prediction_numeric)[0]
        
        st.header("🚀 Prediction Result")
        if prediction_label == "🔴 Contaminated":
            st.error(f"The water is predicted to be: {prediction_label}")
        elif prediction_label == "🟡 Moderate":
            st.warning(f"The water is predicted to be: {prediction_label}")
        else:
            st.success(f"The water is predicted to be: {prediction_label}")
        st.subheader("Input Data Used:")
        st.dataframe(input_df)

# --- TAB 2: SEARCH BY LOCATION (WITH CLEANER DROPDOWNS & CHARTS) ---
with tab2:
    st.header("Search for a Water Sample from the Database")

    states = sorted(df_clean[df_clean['STATE'] != 'Unknown']['STATE'].unique())
    selected_state = st.selectbox("1. Select a State", states)

    state_df = df_clean[df_clean['STATE'] == selected_state]
    locations = sorted(state_df[(state_df['LOCATIONS'] != 'Nan') & (state_df['LOCATIONS'] != 'Unknown')]['LOCATIONS'].unique())
    
    if not locations:
        st.warning("No valid locations found for this state. The data may be missing or swapped.")
    else:
        selected_location = st.selectbox("2. Select a Location (River)", locations)

        location_df = state_df[state_df['LOCATIONS'] == selected_location]
        years = sorted(location_df[location_df['year'] > 0]['year'].unique(), reverse=True)
        
        if not years:
            st.warning("No valid year data found for this location.")
        else:
            selected_year = st.selectbox("3. Select a Year", years)
            
            selected_row_data = location_df[location_df['year'] == selected_year].iloc[0]

            if st.button("Predict for Selected Sample"):
                input_df = selected_row_data[features].to_frame().T
                
                prediction_numeric = pipeline.predict(input_df)
                prediction_label = encoder.inverse_transform(prediction_numeric)[0]
                
                st.header(f"🚀 Prediction for: {selected_location} ({selected_year})")
                if prediction_label == "🔴 Contaminated":
                    st.error(f"The water is predicted to be: {prediction_label}")
                elif prediction_label == "🟡 Moderate":
                    st.warning(f"The water is predicted to be: {prediction_label}")
                else:
                    st.success(f"The water is predicted to be: {prediction_label}")

                st.subheader("Comparison: River Data vs. Safe Limits")
                river_values = [selected_row_data[f] for f in features]
                ideal_ranges = [safe_limits[f]['Ideal'] for f in features]
                polluted_ranges = [safe_limits[f]['Polluted'] for f in features]
                comparison_df = pd.DataFrame({
                    'Parameter': [feature_names[f] for f in features],
                    'River Value': river_values,
                    'Ideal Range (Safe)': ideal_ranges,
                    'Polluted Range': polluted_ranges
                })
                comparison_df['River Value'] = comparison_df['River Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                st.dataframe(comparison_df.set_index('Parameter'))

                # --- 4. Enhanced Visual Profile ---
                st.subheader("Visual Profile of Selected Sample")
                
                phys_features = ['do', 'ph', 'conductivity', 'bod', 'nitrate', 'temp']
                bio_features = ['fecal_coliform', 'total_coliform']
                
                # --- Physicochemical Chart Data ---
                phys_data_list = []
                for f in phys_features:
                    val = selected_row_data[f]
                    phys_data_list.append({
                        'Parameter': feature_names[f],
                        'Value': 0 if pd.isna(val) else val
                    })
                phys_data = pd.DataFrame(phys_data_list)

                # --- Biological Chart Data ---
                bio_data_list = []
                for f in bio_features:
                    val = selected_row_data[f]
                    original_val = 0 if pd.isna(val) else val
                    bio_data_list.append({
                        'Parameter': feature_names[f],
                        'Value': original_val
                    })
                bio_data = pd.DataFrame(bio_data_list)

                # Chart 1: Physicochemical Profile (Bar Chart)
                st.markdown("##### Physicochemical Profile")
                chart1 = alt.Chart(phys_data).mark_bar().encode(
                    x=alt.X('Parameter', sort=None, title=None),
                    y=alt.Y('Value', title='Measured Value'),
                    color=alt.Color('Parameter', legend=None),
                    tooltip=['Parameter', 'Value']
                ).interactive()
                st.altair_chart(chart1, use_container_width=True)

                # Chart 2: Biological Profile (Bar Chart - FIXED)
                st.markdown("##### Biological Profile")
                chart2 = alt.Chart(bio_data).mark_bar().encode(
                    # X-axis plots the Parameter names
                    x=alt.X('Parameter', sort=None, title=None),
                    # Y-axis plots the Value (linear scale)
                    y=alt.Y('Value', title='Measured Value'),
                    # Color the bars by parameter
                    color=alt.Color('Parameter', legend=None),
                    tooltip=['Parameter', 'Value']
                ).interactive()
                st.altair_chart(chart2, use_container_width=True)

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This app uses a Random Forest model to predict water quality. "
    "You can enter values manually or search the 'Indian Water Quality Dataset' "
    "to test real-world samples."
)
