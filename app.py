import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def load_data():
    # Load datasets
    train = pd.read_csv('data/train.csv', parse_dates=['Date'])
    features = pd.read_csv('data/features.csv', parse_dates=['Date'])
    stores = pd.read_csv('data/stores.csv')
    test = pd.read_csv('data/test.csv', parse_dates=['Date'])
    return train, features, stores, test

@st.cache(allow_output_mutation=True)
def preprocess_data(train, features):
    # Merge train and features data on Store and Date
    df = train.merge(features, on=['Store', 'Date'], how='left')

    # Convert IsHoliday from bool to int
    if 'IsHoliday_x' in df.columns and 'IsHoliday_y' in df.columns:
        # Sometimes merge leads to both columns, keep train IsHoliday
        df['IsHoliday'] = df['IsHoliday_x'].fillna(df['IsHoliday_y']).astype(bool).astype(int)
        df.drop(columns=['IsHoliday_x','IsHoliday_y'], inplace=True)
    elif 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(bool).astype(int)
    else:
        # fallback
        df['IsHoliday'] = 0

    # Fill missing MarkDown columns with 0
    for i in range(1,6):
        col = f'MarkDown{i}'
        if col in df.columns:
            df[col].fillna(0, inplace=True)
        else:
            df[col] = 0

    # Extract date features
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    return df

@st.cache(allow_output_mutation=True)
def train_final_model(df):
    # Features used for modeling
    feature_cols = ['Store', 'Dept', 'WeekOfYear', 'Month', 'Year', 'DayOfWeek',
                    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday']

    # Prepare training data
    X = df[feature_cols]
    y = df['Weekly_Sales']

    # Initialize and train the model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

@st.cache(allow_output_mutation=True)
def prepare_test_data(test, features):
    # Merge test and features data
    df_test = test.merge(features, on=['Store', 'Date'], how='left')

    # Convert IsHoliday to int
    if 'IsHoliday_x' in df_test.columns and 'IsHoliday_y' in df_test.columns:
        df_test['IsHoliday'] = df_test['IsHoliday_x'].fillna(df_test['IsHoliday_y']).astype(bool).astype(int)
        df_test.drop(columns=['IsHoliday_x','IsHoliday_y'], inplace=True)
    elif 'IsHoliday' in df_test.columns:
        df_test['IsHoliday'] = df_test['IsHoliday'].astype(bool).astype(int)
    else:
        df_test['IsHoliday'] = 0

    # Fill missing MarkDown columns
    for i in range(1,6):
        col = f'MarkDown{i}'
        if col in df_test.columns:
            df_test[col].fillna(0, inplace=True)
        else:
            df_test[col] = 0

    # Extract date features
    df_test['WeekOfYear'] = df_test['Date'].dt.isocalendar().week.astype(int)
    df_test['Month'] = df_test['Date'].dt.month
    df_test['Year'] = df_test['Date'].dt.year
    df_test['DayOfWeek'] = df_test['Date'].dt.dayofweek

    return df_test

def main():
    st.title("📊 Walmart Sales Forecasting Dashboard")

    st.markdown("""
    This dashboard shows historical sales data and model-based sales forecasts for Walmart stores and departments.
    """)

    # Load data
    train, features, stores, test = load_data()

    # Preprocess train data
    df_train = preprocess_data(train, features)

    # Train final model (cached)
    with st.spinner("Training final model..."):
        model = train_final_model(df_train)

    # Prepare test data
    df_test = prepare_test_data(test, features)

    # Select store and department for visualization
    store_list = sorted(df_train['Store'].unique())
    store_selected = st.sidebar.selectbox("Select Store", store_list)

    dept_list = sorted(df_train[df_train['Store'] == store_selected]['Dept'].unique())
    dept_selected = st.sidebar.selectbox("Select Department", dept_list)

    # Filter historical data for selected store and dept
    hist_data = df_train[(df_train['Store'] == store_selected) & (df_train['Dept'] == dept_selected)].sort_values('Date')

    # Show historical sales plot
    st.subheader(f"Historical Weekly Sales for Store {store_selected}, Dept {dept_selected}")
    plt.figure(figsize=(10,5))
    plt.plot(hist_data['Date'], hist_data['Weekly_Sales'], label='Historical Sales')
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

    # Predict on test for selected store and dept
    test_filtered = df_test[(df_test['Store'] == store_selected) & (df_test['Dept'] == dept_selected)].copy()

    if not test_filtered.empty:
        feature_cols = ['Store', 'Dept', 'WeekOfYear', 'Month', 'Year', 'DayOfWeek',
                        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday']

        X_test = test_filtered[feature_cols]
        test_filtered['Predicted_Weekly_Sales'] = model.predict(X_test)

        st.subheader(f"Forecasted Weekly Sales for Store {store_selected}, Dept {dept_selected}")
        st.dataframe(test_filtered[['Date', 'Predicted_Weekly_Sales']].set_index('Date'))

        # Plot forecast
        plt.figure(figsize=(10,5))
        plt.plot(test_filtered['Date'], test_filtered['Predicted_Weekly_Sales'], label='Forecasted Sales', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Predicted Weekly Sales")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot()
    else:
        st.warning("No test data available for this Store and Department combination.")

if __name__ == "__main__":
    main()

