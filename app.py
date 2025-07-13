import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model

# Load model and transformers
model = load_model('churn_ann_model.h5')
scaler = joblib.load('scaler.pkl')
encoder_gender = joblib.load('encoder_gender.pkl')

# Load Data
df = pd.read_csv('Churn_Modelling.csv')
try:
    df_encoded = df.copy()
    df_encoded['Gender'] = encoder_gender.transform(df_encoded['Gender'])
    df_encoded = pd.get_dummies(df_encoded, columns=['Geography'], drop_first=True)
except Exception as e:
    st.error(f"Encoding failed: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("🧭 Navigation Menu")
option = st.sidebar.radio("📂 Go to Section", [
    "📘 Project Introduction",
    "📊 Visualizations & Insights",
    "🔮 Predict Customer Churn"
])

# 1. Project Introduction
if option == "📘 Project Introduction":
    st.title("💼 Customer Churn Prediction using ANN")

    st.image("customerchurningimage.png", caption="What is Customer Churn?", width=450)

    st.markdown("""
    ### 📌 What is Customer Churn?
    Customer churn refers to the phenomenon where a customer **stops doing business** with a company.

    > 🔁 For a bank: It means the customer **closes their account** or **stops using the bank’s services**.

    ### 🤔 Why Do Customers Churn?
    - Poor customer service  
    - High bank charges  
    - Better offers from competitors  
    - Lack of engagement

    ### ✅ Why Do Some Customers Stay?
    - Strong brand loyalty  
    - Personalized offers  
    - Good interest rates or credit options  
    - Easy-to-use mobile banking

    ---

    ### 🎯 Business Use Case
    A bank wants to proactively predict **which customers are likely to leave**, so they can take actions to retain them.

    > A model is trained to classify whether a customer will **stay (0)** or **churn (1)** based on their profile.

    ---

    ### 📊 Dataset Overview
    The dataset contains **10,000 bank customer records**, including:
    - `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`
    - `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`
    - `EstimatedSalary`
    - **Target**: `Exited` (1 = Churned, 0 = Stayed)

    ---

    ### 🤖 Model: Artificial Neural Network (ANN)
    - We use an **ANN model** built using **TensorFlow** and **Keras**
    - Architecture:  
      `Input Layer → Dense → ReLU Layers → Dropout → Output Layer (Sigmoid)`

    """)

    st.image("annBasicImage.png", caption="Basic ANN Architecture for Churn Prediction", width=450)

    st.markdown("""
    ---

    ### ⚙️ Model Details
    - **Loss Function:** Focal Crossentropy  
    - **Optimizer:** Adam  
    - **Metric Used:** Accuracy

    ---

    ### 🛠️ Technologies Used
    - **Python**, **Pandas**, **NumPy**
    - **Matplotlib**, **Seaborn** for Visualization
    - **Scikit-learn** for Preprocessing & SMOTE
    - **TensorFlow / Keras** for Model Building
    - **Streamlit** for Deployment
    """)
    st.markdown("---")  # Horizontal divider
    # Project Steps – Formatted & Beautiful
    st.subheader("🛠️ Steps Followed to Build This Project")

    with st.expander("📌 Import Libraries and Dataset"):
        st.markdown("""
            ###    
            - Imported necessary libraries  
            - Loaded the dataset
        """)
    with st.expander("📊 Exploratory Data Analysis (EDA)"):
        st.markdown("""
            - Analyzed the dataset  
            - Gathered information about features and target variable  
            - Checked for missing values and data types  
            - Visualized the data to understand customer demographics and churn patterns  
            - Performed univariate & bivariate analysis  
            - Conducted multivariate analysis to identify feature correlations
        """)
    with st.expander("🧹 Data Preprocessing & Preparation "):
        st.markdown("""
            - Dropped unnecessary columns  
            - Handled missing values  
            - Encoded categorical variables  
            - Removed outliers  
            - Separated input and output variables  
            - Split data into training and testing sets  
            - Applied feature scaling using **StandardScaler**  
            - Balanced dataset using **SMOTE (Synthetic Minority Over-sampling Technique)**
        """)
    with st.expander("🤖 Model Building (ANN)"):
        st.markdown("""
            - Used Keras and TensorFlow libraries  
            - Defined layer architecture (input, hidden, output)  
            - Compiled the model using **Focal Loss**, **Adam optimizer**, and **Accuracy** as metric  
            - Trained the model on training set  
            - Visualized training **accuracy** and **loss** curves  
            - Evaluated model on test set
        """)
    with st.expander("💾 Model Saving and Deployment"):
        st.markdown("""
            - Saved the trained model using **joblib**  
            - Built an interactive app using **Streamlit**  
            - Users can input customer details and predict churn  
            - App provides visual insights on customer churn patterns
        """)


    st.markdown("<br><br><br><hr>", unsafe_allow_html=True)
    st.markdown(
    "<center><small>Made with ❤️ by Shiv Sablok | Powered by Streamlit</small></center>",
    unsafe_allow_html=True
)

# 2. Visualizations & Insights
elif option == "📊 Visualizations & Insights":
    st.title("Visual Explorations & Insights")

    # 1. Dataset View
    st.header("1. Dataset View")
    st.dataframe(df.head())

    # 2. Statistical Summary
    st.header("2. Statistical Summary")
    st.dataframe(df.describe())
    st.markdown("""
    - **CreditScore:** Most customers have a good score between 600 and 700.
    - **Age:** Wide age range, with some older customers.
    - **Balance:** Many customers have zero balance; some hold very high balances.
    - **EstimatedSalary:** Almost uniform distribution.
    """)

    # 3. Univariate Analysis
    st.header("3. Univariate Analysis")
    st.subheader("Categorical Features")
    categorical_cols = ['Gender', 'Geography', 'Exited', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Tenure']
    sns.set(style='whitegrid')
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df, x=col, palette='Set2', ax=ax)
        ax.set_title(f'Countplot of {col}')
        st.pyplot(fig)
    st.markdown("""
    **Insights:**
    - **Gender:** Slightly more male customers.
    - **Geography:** France dominates, Germany has higher churn.
    - **Exited:** Shows imbalance.
    - **HasCrCard/IsActiveMember:** Most have cards and are active.
    - **NumOfProducts:** Majority have 1 product.
    - **Tenure:** Fairly uniform.
    """)

    st.subheader("Numerical Features")
    numeric_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[col], kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)
    st.markdown("""
    **Insights:**
    - **CreditScore:** Normal distribution centered ~650.
    - **Age:** Skewed toward younger.
    - **Balance:** High number of zeros.
    - **EstimatedSalary:** Uniform spread.
    """)

    # 4. Bivariate Analysis
    st.header("4. Bivariate Analysis")
    st.subheader("Categorical vs Exited")
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df, x=col, hue='Exited', palette='Set1', ax=ax)
        ax.set_title(f'{col} vs Exited')
        st.pyplot(fig)
    st.markdown("""
    **Insights:**
    - **Germany, Females, and inactive members churn more.**
    """)

    st.subheader("Numerical vs Exited")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(data=df, x='Exited', y=col, palette='Set1', ax=ax)
        ax.set_title(f"{col} vs Exited")
        st.pyplot(fig)
    st.markdown("""
    **Insights:**
    - **Age and Balance are significantly higher for churned customers.**
    """)

    # 5. Multivariate Analysis
    st.header("5. Multivariate Analysis - Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_encoded.select_dtypes(include='number').corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.markdown("""
    **Insights:**
    - Age and Balance show higher correlation with Churn.
    - CreditCard and EstimatedSalary show low correlation.
    """)

    # 6. Dataset after Encoding
    st.header("6. Encoded Dataset Preview")
    st.dataframe(df_encoded.head())

    # 7. Class Imbalance
    st.header("7. Class Imbalance")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(x='Exited', data=df, palette='Set1', ax=ax)
    ax.set_title("Class Imbalance (Exited)")
    st.pyplot(fig)
    st.markdown("""
    - **Imbalance Notice:** Only ~20% customers churned.
    """)

    # 8. Balanced Dataset (Post-SMOTE)
    st.header("8. Balanced Dataset After SMOTE")
    smote_counts = pd.Series([8000, 8000], index=['Stayed (0)', 'Churned (1)'])
    fig, ax = plt.subplots(figsize=(4, 3))
    smote_counts.plot(kind='bar', color=['lightgreen', 'tomato'], ax=ax)
    ax.set_title("SMOTE Balanced Class Distribution")
    st.pyplot(fig)

    # 9. ANN Model Training Curves
    st.header("9. ANN Model Training Curves")
    st.image('accuracyTraining.png', caption='📈 Training Accuracy', width=500)
    st.image('lossTraining.png', caption='📉 Training Loss', width=500)

    # 10. Final Model Accuracy
    st.header("🔍✨ 10. Final Model Performance")
    st.markdown("""
    <div style='border: 2px solid #4CAF50; padding: 15px; border-radius: 10px;'>
        <h4 style='color: #2E8B57;'>📌 Final Evaluation Summary:</h4>
        <ul style='list-style-type: none; padding-left: 10px;'>
            <li><strong>✅ Training Accuracy:</strong> 83.50%</li>
            <li><strong>📉 Training Loss:</strong> 9.50%</li>
            <li><strong>🧪 Test Accuracy:</strong> 80.48%</li>
            <li><strong>❌ Test Loss:</strong> 11.21%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# 3. Prediction Interface
elif option == "🔮 Predict Customer Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("""
    Use the form below to enter customer details. Our ANN model will analyze the inputs and predict the likelihood of the customer churning.
    """)

    st.markdown("---")

    # Form Inputs with Tooltips
    credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650, help="A measure of creditworthiness. Lower scores may increase churn risk.")
    geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"], help="Customer's country of residence. Churn behavior varies across regions.")
    gender = st.selectbox("👤 Gender", ["Male", "Female"], help="Some gender-based patterns may influence churn probability.")
    age = st.slider("🎂 Age", min_value=18, max_value=92, value=35, help="Older customers may exhibit different churn patterns.")
    tenure = st.slider("🏦 Tenure (Years with Bank)", 0, 10, 3, help="Longer-tenured customers often have stronger loyalty.")
    balance = st.number_input("💰 Account Balance", min_value=0.0, value=10000.0, help="Account balance may impact the customer's financial engagement.")
    num_products = st.selectbox("🛍️ Number of Products", [1, 2, 3, 4], help="How many banking products the customer uses.")
    has_card = st.selectbox("💳 Has Credit Card?", ["Yes", "No"], help="Whether the customer owns a credit card from the bank.")
    is_active = st.selectbox("📶 Is Active Member?", ["Yes", "No"], help="Whether the customer actively engages with the bank.")
    est_salary = st.number_input("📈 Estimated Salary", min_value=0.0, value=50000.0, help="Customer's income estimate. Might reflect financial capability.")

    st.markdown("""
    > 📌 Click **Predict** below to see if the customer is likely to churn or stay.
    """)

    if st.button("🚀 Predict"):
        geo_map = {'France': [0, 0], 'Germany': [1, 0], 'Spain': [0, 1]}
        geography_encoded = geo_map[geography]
        gender_encoded = encoder_gender.transform([gender])[0]
        has_card = 1 if has_card == 'Yes' else 0
        is_active = 1 if is_active == 'Yes' else 0

        input_data = np.array([[credit_score, gender_encoded, age, tenure, balance,
                                num_products, has_card, is_active, est_salary] + geography_encoded])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.subheader("🔎 Prediction Result")
        st.write(f"Model output (probability of churn): **{prediction[0][0]:.4f}**")

        if prediction[0][0] > 0.5:
            st.error("⚠️ The customer is **likely to churn**. Consider proactive retention strategies.")
        else:
            st.success("✅ The customer is **likely to stay**. Maintain engagement.")
