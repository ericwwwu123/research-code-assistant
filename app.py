import streamlit as st
import pandas as pd
import requests
import pyreadstat
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import re
import ast


st.set_page_config(page_title="Research Code Assistant")
st.title("Research Code Assistant")


# 🔀 Local mode toggle
st.markdown("Supports code translation, explanation, and safe data analysis across common research file formats.")

api_key = os.getenv("DEEPSEEK_API_KEY")  # Replace with your DeepSeek API key
url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def load_uploaded_file(uploaded_file, key_prefix="default"):
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "sav":
        df, meta = pyreadstat.read_sav(uploaded_file)
    elif file_type == "xlsx":
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select sheet:", xls.sheet_names, key=f"{key_prefix}_sheet")
        df = pd.read_excel(xls, sheet_name=sheet_name)
    else:
        st.error("Unsupported file type.")
        return None
    return df

# Shared function for sensitive field masking
def scan_and_mask_sensitive_fields(df, key_prefix="default"):
    st.subheader("🔐 Sensitive Field Scanner")
    all_columns = list(df.columns)
    sensitive_keywords = ["name", "id", "email", "phone", "dob", "birth", "address", "ssn", "student", "gender"]
    suggested_sensitive_cols = [col for col in df.columns if any(key in col.lower() for key in sensitive_keywords)]

    confirmed_sensitive_cols = st.multiselect(
        "Select fields that may contain sensitive info:",
        options=all_columns,
        default=suggested_sensitive_cols,
        key=f"{key_prefix}_sensitive_multiselect"
    )

    if confirmed_sensitive_cols:
        masked_df = df.copy()
        for col in confirmed_sensitive_cols:
            masked_df[col] = masked_df[col].astype(str).apply(lambda x: "****" if len(x) > 0 else x)

        st.markdown("### Preview of Masked Data (first 10 rows shown below)")
        st.dataframe(masked_df.head(10))

        with st.expander("🔍 View original data sample (for debug use only)"):
            st.dataframe(df.head(10))

        st.markdown("⚠️ Only the **masked** first 10 rows will be sent to the AI model to generate code. Your raw data will never leave your machine.")

        st.subheader("📊 Descriptive Statistics")
        st.dataframe(df.describe(include='all').transpose())

        st.subheader("📦 Download Masked Data")
        csv = masked_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="masked_data.csv">Download Cleaned CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # 👇 插入以下新功能模块

        with st.expander("🧬 Column Type Summary"):
            dtypes = pd.DataFrame(df.dtypes, columns=["dtype"])
            st.dataframe(dtypes)

        with st.expander("🧪 Missing Value Analysis"):
            missing = df.isnull().sum()
            missing_filtered = missing[missing > 0]

            if not missing_filtered.empty:
                st.write("📉 Missing Value Summary")
                fig, ax = plt.subplots()
                missing_filtered.sort_values(ascending=False).plot(kind='bar', ax=ax)
                st.pyplot(fig)
            else:
                st.info("✅ No missing values found in the dataset.")

        return masked_df
    else:
        st.info("Please select at least one sensitive field to mask before continuing.")
        return None

# Tabs
options = ["🔒 Privacy", "📁 Offline Tools"]
if st.session_state.get("use_api"):
    options.append("🌐 Online Tools")
options = ["🔒 Privacy", "📁 Offline Tools"]
labels = {
    "🔒 Privacy": "🔒 Privacy",
    "📁 Offline Tools": "📁 Offline Tools",
    "🌐 Online Tools": "🌐 Online Tools" if st.session_state.get("use_api", False) else "❌ Online Tools (Enable AI)"
}
options.append("🌐 Online Tools")  # 始终保留在线工具入口

selected = st.sidebar.radio("🔁 Choose Interface Mode:", options, format_func=lambda x: labels[x])
mode = selected


if mode == "🔒 Privacy":
    if "use_api" not in st.session_state:
        st.session_state["use_api"] = False

    st.subheader("🌐 AI Model Access")
    st.session_state["use_api"] = st.toggle(
        "Enable AI model (DeepSeek)",
        value=st.session_state["use_api"],
        help="Turn off to run in local-only mode without sending data to external APIs."
    )

    if st.session_state["use_api"]:
        st.success("✅ AI-powered features are enabled. You can now use Code Translator, Code Explainer, and Data Analyzer.")
    else:
        st.info("🧠 AI features are currently disabled. You may continue using offline tools only.")
    st.subheader("🔐 Data Privacy & Local Mode")
    st.markdown("""
This AI Agent is designed with research data privacy in mind:

- **Local Mode Available:** You can disable all external API calls using the toggle at the top.
- **No Data Uploads by Default:** All uploaded files are processed in memory only.
- **Sample-Only AI Usage:** When enabled, the AI model receives only column names and 10 sample rows (not full dataset).
- **Temporary Memory Use:** Uploaded data is not saved to disk unless explicitly exported.
- **You’re In Control:** All code is shown before execution. You may copy and run it externally (e.g. Jupyter).

This ensures that sensitive research data remains protected, whether working online or offline.
""")

elif mode == "📁 Offline Tools":
    st.header("📁 Offline Tools")
    uploaded_file = st.file_uploader("Upload a dataset (.csv, .xlsx, .sav):", type=["csv", "xlsx", "sav"], key="offline_upload")
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]
        try:
            df = load_uploaded_file(uploaded_file, key_prefix="offline")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            df = None
        if df is not None:
            st.success("✅ File loaded locally")
            masked_df = scan_and_mask_sensitive_fields(df, key_prefix="offline")
            with st.expander("📝 Analysis Summary Generator"):
                selected_sections = st.multiselect(
                    "Select sections to include in the report:",
                    ["Dataset Overview", "Variable Types", "Missing Values", "Descriptive Statistics", "Group Comparison"],
                    default=["Dataset Overview", "Descriptive Statistics"]
                )

                summary_md = ""

                if "Dataset Overview" in selected_sections:
                    summary_md += f"### Dataset Overview\n\n- Number of rows: {df.shape[0]}\n- Number of columns: {df.shape[1]}\n\n"

                if "Variable Types" in selected_sections:
                    types = df.dtypes.value_counts().to_dict()
                    summary_md += "### Variable Types\n\n"
                    for t, count in types.items():
                        summary_md += f"- {count} {t} variables\n"
                    summary_md += "\n"

                if "Missing Values" in selected_sections:
                    missing = df.isnull().sum()
                    total_missing = (missing > 0).sum()
                    summary_md += f"### Missing Value Summary\n\n- Variables with missing values: {total_missing}\n\n"

                if "Descriptive Statistics" in selected_sections:
                    summary_md += "### Descriptive Statistics\n\nStandard descriptive stats have been generated and reviewed.\n\n"

                if "Group Comparison" in selected_sections:
                    summary_md += "### Group Comparison\n\nComparative analysis between selected groups has been performed. See visualizations above.\n\n"

                st.markdown("#### 📄 Generated Markdown Report")
                st.code(summary_md, language="markdown")

                st.download_button("Download .md", summary_md, file_name="summary_report.md")

            with st.expander("🧽 Fill Missing Values"):
                fill_cols = st.multiselect("Choose columns to fill:", df.columns[df.isnull().any()])
                fill_method = st.radio("Fill method:", ["Mean", "Median"])
                if st.button("Apply Fill"):
                    df_filled = df.copy()
                    for col in fill_cols:
                        if fill_method == "Mean":
                            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                        else:
                            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                    st.success("Missing values filled.")
                    st.dataframe(df_filled.head())

            with st.expander("📦 Boxplot Visualization"):
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                col_to_plot = st.selectbox("Select column for boxplot:", numeric_cols)
                if col_to_plot:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col_to_plot], ax=ax)
                    st.pyplot(fig)
        

elif mode == "🌐 Online Tools":
    if not st.session_state.get("use_api", False):
        st.warning("🔒 Online Tools are disabled. Please enable the AI model in the Privacy tab to use this feature.")
        st.stop()
    st.markdown("AI-powered tools are available below only if model is enabled.")
    tab1, tab2, tab3 = st.tabs(["📊 Data Analyzer", "🔄 Code Translator", "💬 Code Explainer"])

    with tab1:
        if not st.session_state.get("use_api"):
            st.warning("🔒 This feature is disabled in local mode. Enable the AI model (DeepSeek) to use Data Analyzer.")
            st.stop()

        st.subheader("Upload dataset for analysis and sensitive data scanning")
        uploaded_file = st.file_uploader("Upload your dataset (.csv, .xlsx, .sav):", type=["csv", "xlsx", "sav"], key="online_data")

        if uploaded_file:
            file_type = uploaded_file.name.split(".")[-1]
            try:
                df = load_uploaded_file(uploaded_file, key_prefix="analyzer")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                df = None 

            if df is not None:
                st.success("✅ File uploaded")
                masked_df = scan_and_mask_sensitive_fields(df, key_prefix="analyzer")
                if masked_df is not None:
                    st.subheader("Describe Your Analysis Goal")
                    user_goal = st.text_area("e.g., Compare BMI across education levels")
                    if st.button("Generate Analysis Code"):
                        sample_text = masked_df.head(10).to_csv(index=False)
                        prompt = f"""
You are a data analyst. Based on the following dataset structure and user's goal, generate appropriate Python code.

⚠️ Do NOT load any dataset from a CSV file.
The data is already available in a Pandas DataFrame named `df`.

User goal: {user_goal}

Sample:
```csv
{sample_text}
```
"""
                        payload = {
                            "model": "deepseek-coder",
                            "messages": [
                                {"role": "system", "content": "You are a helpful data science assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.3
                        }
                        try:
                            response = requests.post(url, headers=headers, json=payload)
                            if response.status_code == 200:
                                code = response.json()["choices"][0]["message"]["content"]
                                st.code(code, language="python")
                        except Exception as e:
                            st.error(f"Generation failed: {e}")

    with tab2:
        if not st.session_state.get("use_api"):
            st.warning("🔒 Translation disabled. Enable AI model.")
            st.stop()
        st.subheader("Code Translator (Bidirectional)")
        source_lang = st.selectbox("Source language:", ["Python", "R", "SAS", "Stata", "SPSS"])
        target_lang = st.selectbox("Target language:", ["Python", "R", "SAS", "Stata", "SPSS"])
        source_code = st.text_area("Paste source code:")
        if st.button(f"Translate from {source_lang} to {target_lang}"):
            user_msg = f"""Translate this {source_lang} code into {target_lang} with clear comments:
```{source_lang.lower()}
{source_code}
```"""
            payload = {
                "model": "deepseek-coder",
                "messages": [
                    {"role": "system", "content": "You are an expert programmer."},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.2
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                st.code(result, language="python")

    with tab3:
        if not st.session_state.get("use_api"):
            st.warning("🔒 Explanation disabled. Enable AI model.")
            st.stop()
        st.subheader("Explain Python Code with Familiar Concepts")
        familiar_lang = st.selectbox("Your familiar language:", ["R", "SAS", "Stata", "SPSS"])
        code = st.text_area("Paste Python code:")
        if st.button("Explain"):
            user_msg = f"""Explain this Python code with comparisons to {familiar_lang}:
        ```python
        {code}
        ```"""
            payload = {
                "model": "deepseek-coder",
                "messages": [
                    {"role": "system", "content": "You are a programming tutor."},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.3
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                explanation = response.json()["choices"][0]["message"]["content"]
                st.markdown(explanation)
