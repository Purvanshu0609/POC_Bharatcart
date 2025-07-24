import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Streamlit UI Elements
st.title("BharatCart Inventory & Retention Analysis")

# Load the dataset
@st.cache
def load_data():
    # Ensure the file path is correct on your machine
    file_path = r"C:\Users\purvanshu\Downloads\BharatCart_Ecommerce_Dataset.csv"
    df = pd.read_csv(file_path)
    
    # Convert the 'date_' column to datetime format
    df['date_'] = pd.to_datetime(df['date_'], format='%d-%m-%Y', errors='coerce')  # Handle any parsing errors
    return df
    
def load_data2():
    # Ensure the file path is correct on your machine
    file_path = r"C:\Users\purvanshu\Exercise\encoded_bharatcart_data.csv"
    df2 = pd.read_csv(file_path)
    
    # Convert the 'order_date' column to datetime format
    df2['order_date'] = pd.to_datetime(df2['order_date'], format='%Y-%m-%d', errors='coerce')
    return df2

# Data Cleaning Process
def clean_data(df):
    # Drop rows with critical missing data (if applicable)
    df_cleaned = df.dropna(subset=['procured_quantity', 'unit_selling_price', 'age', 'stock_level'])

    # Fill missing values for numerical columns (excluding datetime columns)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

    # Drop duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    return df_cleaned

# Load and clean the data
df = load_data()
df2 = load_data2()
df_cleaned = clean_data(df)
df2_cleaned = clean_data(df2)

# Page selection
page = st.radio("Select a Page", ['Home', 'Model Evaluation'])

if page == 'Home':
    # Project Description
    st.markdown("""
    # Project Title: Solving Inventory Gaps, Customer Segmentation & Retention Using Data Analysis and ML
    *Problem Statement*: BharatCart is facing inventory imbalances, lacks customer segmentation, and struggles with low retention, limiting its growth.

    *Objective*: To analyse and address key e-commerce challenges faced by BharatCart namely inventory imbalances, lack of customer segmentation, and low customer retention using data analysis and machine learning.
    """)

    # Display basic data information
    st.subheader("Dataset Overview")
    st.write(f"Shape of the data: {df.shape}")
    st.write("Data types and missing values:")
    st.write(df.dtypes)
    st.write("Missing values count:")
    st.write(df.isnull().sum())

    # EDA: Visualizing Distributions
    st.subheader("Data Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df_cleaned['procured_quantity'], kde=True, ax=ax[0])
    ax[0].set_title("Procured Quantity Distribution")
    sns.histplot(df_cleaned['unit_selling_price'], kde=True, ax=ax[1])
    ax[1].set_title("Unit Selling Price Distribution")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Matrix")
    numeric_columns = df_cleaned.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Category Breakdown
    st.subheader("Breakdown by City and Product Category")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(x='city_name', data=df_cleaned, ax=ax[0])
    ax[0].set_title("Orders by City")
    sns.countplot(x='product_category', data=df_cleaned, ax=ax[1])
    ax[1].set_title("Orders by Product Category")
    st.pyplot(fig)

    # Inventory Imbalance
    st.subheader("Inventory Imbalance Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='stock_level', y='procured_quantity', data=df_cleaned, ax=ax)
    ax.set_title("Stock Level vs. Procured Quantity")
    st.pyplot(fig)

    # City filter
    city_filter = st.selectbox("Select City", df_cleaned['city_name'].unique())
    filtered_data = df_cleaned[df_cleaned['city_name'] == city_filter]
    st.subheader(f"Data for {city_filter}")
    st.write(filtered_data.head())

    # Product category filter
    product_category_filter = st.selectbox("Select Product Category", df_cleaned['product_category'].unique())
    filtered_by_category = df_cleaned[df_cleaned['product_category'] == product_category_filter]
    st.subheader(f"Filtered Data for {product_category_filter}")
    st.write(filtered_by_category.head())

    # Customer Segmentation
    st.subheader("Customer Segmentation: Age vs. Income Bracket")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='income_bracket', y='age', data=df_cleaned, ax=ax)
    ax.set_title("Age Distribution by Income Bracket")
    st.pyplot(fig)

    # Show cleaned data
    st.subheader("Cleaned Dataset")
    st.write(df_cleaned.head())

elif page == 'Model Evaluation':
    # Display Evaluation Metrics
    st.subheader("Logistic Regression Model Evaluation")
    
    # Define target column
    target_column = 'stock_deficit_flag'
    
    # Prepare features - exclude datetime columns
    X = df2_cleaned.select_dtypes(exclude=['datetime64']).drop(columns=[target_column])
    y = df2_cleaned[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit pipeline
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        LogisticRegression(max_iter=1000)
    )
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred_log = pipeline.predict(X_test)
    y_prob_log = pipeline.predict_proba(X_test)[:, 1]
    
    # Display Evaluation Metrics
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred_log)
    st.write(cm)
    
    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred_log, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    
    st.write(f"**Training Accuracy:** {pipeline.score(X_train, y_train):.4f}")
    st.write(f"**Test Accuracy:** {pipeline.score(X_test, y_test):.4f}")
    st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_prob_log):.4f}")
    st.write(f"**Balanced Accuracy Score:** {balanced_accuracy_score(y_test, y_pred_log):.4f}")

    # Feature Importance (for logistic regression)
    if hasattr(pipeline.named_steps['logisticregression'], 'coef_'):
        st.subheader("Feature Importance")
        coefficients = pipeline.named_steps['logisticregression'].coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(coefficients)
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)