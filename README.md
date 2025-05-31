# Walmart Sales Forecasting Project

## Overview

This project focuses on forecasting department-wide weekly sales for 45 Walmart stores using historical sales data, store features, regional economic indicators, and promotional markdown events. The goal is to build accurate predictive models, analyze the impact of holidays and markdowns on sales, and deploy an interactive dashboard for visualization.

## Project Scope

- Predict weekly sales by store and department using various machine learning and time series models.
- Model the effects of promotional markdowns and holiday periods, which have a greater impact on sales.
- Compare the performance of models such as Linear Regression, Random Forest, XGBoost, ARIMA, and Prophet.
- Deploy an interactive Streamlit dashboard to visualize historical sales and forecasted sales.

## Dataset Description

The dataset includes the following files:

- `stores.csv`: Information about the 45 Walmart stores – type and size.
- `train.csv`: Historical training data including store, department, date, weekly sales, and holiday flags.
- `test.csv`: Test data with store, department, and dates for which sales need to be predicted.
- `features.csv`: Additional regional and marketing data including Temperature, Fuel Price, CPI, Unemployment, and Markdown variables.

## Project Structure
walmart_sales_forecasting/\n 
├── data/ # Contains raw dataset files (.csv)\n 
├── notebooks/ # Jupyter notebooks for EDA, modeling \n
├── src/ # Source code & scripts (e.g., preprocessing, modeling)\n 
├── app.py # Streamlit dashboard app \n
├── requirements.txt # Python dependencies file (optional) \n
└── README.md # This project README\n



## Setup and Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/iqranizam/walmart_sales_forecasting.git
   cd walmart_sales_forecasting

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
4. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
6. Download the Walmart dataset from Kaggle and place the CSV files inside the data/ folder
7. **Run the Streamlit dashboard**:
   ```bash
   streamlit run app.py
