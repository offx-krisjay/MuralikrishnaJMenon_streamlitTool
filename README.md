# Data Analytics & ML Streamlit App

A web-based application for **data cleaning, profiling, and simple machine learning predictions**. Built with **Streamlit**, **Polars**, and **Pandas**, this tool is designed to handle large CSV datasets efficiently.

---

## 🚀 Features

### 1. Data Loading & Cleaning
- Upload CSV files of significant size (100 MB+).
- Utilizes **Polars** with lazy operations for speed; falls back to **Pandas** when required.
- Data cleaning operations include:
  - Dropping `NaN` values
  - Removing duplicates
  - Type conversions
  - Normalization / scaling
  - Filtering rows based on conditions

### 2. Data Profiling
- Integrated with **ydata-profiling** (or `pandas-profiling`) for comprehensive data analysis.
- Provides:
  - Statistics summary
  - Correlations
  - Missing value insights
  - Warnings and suggestions
- Supports **Polars** dataframes by converting to Pandas (`.to_pandas()`).

### 3. Interactive UI
- Built with **Streamlit** widgets:
  - File upload
  - Operation selection
  - Live previews of data
  - Download cleaned CSV
- Designed to handle large datasets gracefully.

### 4. ML Prediction Module (Add-on)
- Users can upload/train a simple ML model or load a pre-trained one.
- Input values via the UI to get predictions:
  - Classification of uploaded rows
  - Predictions for user-provided values

### 5. Deployment
- Hosted live on **Streamlit Cloud/Share**.
- Optimized for concurrent users with:
  - Lazy loading
  - Profiling caching
  - Efficient memory handling

---

## 📦 Requirements

- Python >= 3.9
- `streamlit`
- `polars`
- `pandas`
- `ydata-profiling`
- `scikit-learn`
- `numpy`
