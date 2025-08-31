import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import streamlit.components.v1 as components

st.title("Data-Cleansing, Profiling & ML Tool")
st.subheader("By Muralikrishna")

up_file = st.file_uploader("Upload CSV", type=['csv'])
if up_file is not None:
    pl_file = pl.read_csv(up_file)
    st.toast("Data Loaded successfully")
    st.expander("Top 10 Rows").write(pl_file.head(10))
    st.expander("Bottom 10 Rows").write(pl_file.tail(10))

    df = pl_file.to_pandas()

    st.sidebar.header("Data Cleaning Operations")
    if st.sidebar.checkbox("Drop NaN values"):
        df.dropna(inplace=True)
    if st.sidebar.checkbox("Remove Duplicate rows"):
        df.drop_duplicates(inplace=True)
    if st.sidebar.checkbox("Normalize numeric columns"):
        scaler = MinMaxScaler()
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = scaler.fit_transform(df[num_cols])
    if st.sidebar.checkbox("Type Conversion"):
        col_conv = st.sidebar.selectbox("Select column to convert", df.columns)
        new_dtype = st.sidebar.selectbox("Select new type", ["int", "float"])
        try:
            df[col_conv] = df[col_conv].astype(new_dtype)
        except Exception as e:
            st.error(f"Conversion caused an error: {e}")
    if st.sidebar.checkbox("Filter rows"):
        filt_col = st.sidebar.selectbox("Select column to filter", df.columns)
        filt_val = st.sidebar.text_input("Enter filter value")
        if filt_val:
            df = df[df[filt_col].astype(str).str.contains(filt_val)]

    st.subheader("Cleaned Data Preview")
    if df.empty:
        st.warning("DataFrame is empty after cleaning.")
    else:
        st.dataframe(df.head(100))
    clean_df=df
    st.sidebar.download_button("Cleaned Data",clean_df.to_csv(index=False),"cleaned_data.csv")

    st.subheader("Profiling Report")
    if st.checkbox("Generate profiling report"):
        if df.empty or df.shape[1] == 0:
            st.warning("DataFrame is empty. Cannot generate profiling report.")
        else:
            prof_rep = ProfileReport(df, explorative=True)
            components.html(prof_rep.to_html(), height=1000, scrolling=True)

    
    st.subheader("Encoding")
    if st.checkbox("Convert all String columns to numerical value"):
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        enc_type = st.selectbox("Encoding type", ["Label Encoding", "One-Hot Encoding"])
        if st.button("Encode"):
            if enc_type == "Label Encoding":
                le = LabelEncoder()
                for col in cat_cols:
                    df[col + "_enc"] = le.fit_transform(df[col])
                df.drop(columns=cat_cols, inplace=True)
            else:
                df = pd.get_dummies(df, columns=cat_cols)

    st.subheader("Encoded Data Preview")
    st.dataframe(df.head(100))


    st.subheader("Machine Learning")
    ml_type = st.selectbox("Type", ["Regression", "Classification"])
    features = st.multiselect("Select Features", df.columns)
    target = st.selectbox("Select target", df.columns)

    if features and target:
        X, y = df[features], df[target]
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

        if ml_type == "Regression":
            model = RandomForestRegressor()
            model.fit(xtrain, ytrain)
            rpred = model.predict(xtest)
            st.write("R2 Score:", r2_score(ytest, rpred))
            st.write("Mean squared error:", mean_squared_error(ytest, rpred))
        else:
            model = RandomForestClassifier()
            model.fit(xtrain, ytrain)
            cpred = model.predict(xtest)
            st.write("Accuracy:", accuracy_score(ytest, cpred))
            st.text(classification_report(ytest, cpred))

        st.subheader("User Input Prediction")
        ip_pred = st.radio("Predict using:", ["Manual Input", "Upload File"])
        if ip_pred == "Manual Input":
            input_df = {}
            for col in features:
                val = st.text_input(f"Value for {col}")
                input_df[col] = float(val) if val else 0
            if st.button("Predict"):
                inp_pred = model.predict(pd.DataFrame([input_df]))
                st.success(f"Prediction: {inp_pred[0]}")
        else:
            us_file = st.file_uploader("Upload new CSV for prediction", type=["csv"])
            if us_file:
                us_df = pd.read_csv(us_file)
                us_pred = model.predict(us_df[features])
                us_df['Predictions'] = us_pred
                st.dataframe(us_df)
                    st.download_button(
                        "Download predictions for user file",
                        us_df.to_csv(index=False),
                        "user_file_predictions.csv",
                        "text/csv"
                    )





