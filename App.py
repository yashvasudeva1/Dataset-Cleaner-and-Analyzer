import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title(":material/folder: Dataset Cleaner and Analyser")
st.write("This app helps you in making your dataset cleaner, outlier free and ready for training")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

file = st.file_uploader("")
if file is not None:
    df = load_data(file)
    # Initialize cleaned_df in session_state
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = df.copy()

    st.write("Preview of your dataset:")
    st.dataframe(st.session_state["clean_df"], use_container_width=True)
    
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Analysis", "Visualisation", "Facts", "Outliers", "Predictor", "Distribution"]
    )

    with tab0:
        st.write(st.session_state["clean_df"].describe())

    with tab1:
        with st.container(border=True):
            numeric_columns = st.session_state["clean_df"].select_dtypes(include='number').columns.tolist()
            selected_columns = st.multiselect("Columns", numeric_columns, default=numeric_columns)
            if selected_columns:
                st.line_chart(st.session_state["clean_df"][selected_columns], height=250, use_container_width=True)
            else:
                st.info("Please select at least one column to display the chart.")
    with tab2:
        def dataset_facts(df):
            st.header("🧩 Facts About Your Dataset")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
            st.subheader("🥇 General Information")
            st.write(f"**Total missing cells:** {df.isnull().sum().sum()}")
            st.write(f"**Columns with missing values:** {[c for c in df.columns if df[c].isnull().any()]}")
        
            st.subheader("📊 Column-specific Facts")
            for col in df.columns:
                st.markdown(f"**• {col}**")
                n_missing = df[col].isnull().sum()
                st.write(f"- Missing: {n_missing} ({n_missing/len(df)*100:.1f}%)")
        
                n_unique = df[col].nunique(dropna=True)
                st.write(f"- Unique values: {n_unique}")
        
                if pd.api.types.is_numeric_dtype(df[col]):
                    minv = df[col].min()
                    maxv = df[col].max()
                    meanv = df[col].mean()
                    st.write(f"- Min: {minv}, Max: {maxv}, Mean: {meanv:.2f}")
                    mostval = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                    st.write(f"- Most frequent: {mostval}")
                else:
                    mostval = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                    top_count = df[col].value_counts().max()
                    st.write(f"- Most frequent: '{mostval}' ({top_count} times)")
                    if n_unique < 10:
                        st.write(f"- All unique: {df[col].unique()}")
        
            st.subheader("🎯 Interesting Numeric Facts")
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                max_sum_col = df[numeric_cols].sum().idxmax()
                st.write(f"- Column with highest total sum: **{max_sum_col}**")
                most_var_col = df[numeric_cols].std().idxmax()
                st.write(f"- Most variable column: **{most_var_col}** (std: {df[most_var_col].std():.2f})")
            else:
                st.write("- No numeric columns to summarize.")

    with tab3:
        columns = st.session_state["clean_df"].select_dtypes(include=[np.number]).columns
        # Initial outlier report (before removal)
        outlier_report = []
        for col in columns:
            q1, q3 = st.session_state["clean_df"][col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            n_outliers = ((st.session_state["clean_df"][col] < lower) | (st.session_state["clean_df"][col] > upper)).sum()
            outlier_report.append({"Column Name": col, "Number of Outliers": n_outliers})
        outliers = pd.DataFrame(outlier_report)
        st.write("Current outliers:")
        st.write(outliers)

        remove_outlier = st.button("Remove the Outliers")
        if remove_outlier:  # Clean data and update session_state
            temp_df = st.session_state["clean_df"]
            for col in columns:
                q1, q3 = temp_df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                temp_df = temp_df[(temp_df[col] >= lower) & (temp_df[col] <= upper)]
            st.session_state["clean_df"] = temp_df  # Update the cleaned DataFrame

            # Report remaining outliers
            outlier_report = []
            for col in columns:
                q1, q3 = st.session_state["clean_df"][col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_outliers = ((st.session_state["clean_df"][col] < lower) | (st.session_state["clean_df"][col] > upper)).sum()
                outlier_report.append({"Column Name": col, "Number of Outliers": n_outliers})
            outliers = pd.DataFrame(outlier_report)
            st.write("Remaining outliers after removal:")
            st.write(outliers)
            st.subheader("Cleaned Dataset")
            st.write(temp_df)

    with tab5:
        numeric_cols = st.session_state["clean_df"].select_dtypes(include=np.number).columns
        for col in numeric_cols:
            st.subheader(f"Histogram and KDE for {col}")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state["clean_df"][col].dropna(), kde=True, bins=20, color="grey", ax=ax)
            ax.set_title(f"Histogram and KDE of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close(fig)
