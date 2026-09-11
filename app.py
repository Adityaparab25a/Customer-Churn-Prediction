import streamlit as st
import pandas as pd
import joblib

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Load trained model
model = joblib.load("customer_churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction System")

st.markdown("### Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "78.39%")

with col2:
    st.metric("Recall", "68.45%")

with col3:
    st.metric("F1 Score", "62.75%")

with col4:
    st.metric("ROC-AUC", "83.59%")

st.write(
    "Use customer information to estimate churn probability "
    "and identify customers who may need retention support."
)

# Customer details

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

st.subheader("Services")

col1, col2 = st.columns(2)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

st.subheader("Account Information")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.subheader("Charges & Tenure")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "Predict Churn",
        use_container_width=True
    )

if predict_button:

    # Convert Senior Citizen selection to 0/1
    senior_citizen_value = 1 if senior_citizen == "Yes" else 0

    # Create input DataFrame
    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen_value],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Get churn probability
    probability = model.predict_proba(customer)[0][1]

    # Our chosen threshold
    threshold = 0.40

    # Save prediction to history
    prediction_result = "High Risk" if probability >= threshold else "Low Risk"

    st.session_state.prediction_history.append({
        "Churn Probability": f"{probability * 100:.2f}%",
        "Risk Level": prediction_result,
        "Tenure": f"{tenure} months",
        "Monthly Charges": f"${monthly_charges:.2f}",
        "Contract": contract
    })

    st.subheader("Prediction Result")
    
    st.metric(
    "Churn Probability",
    f"{probability * 100:.2f}%"

    )
    st.progress(
    min(float(probability), 1.0),
    text=f"Churn probability: {probability * 100:.2f}%" 

    )
    if probability >= threshold:
        st.error("HIGH CHURN RISK")
        st.write(
            "This customer has a higher likelihood of churning. "
            "Consider taking retention action."
        )

        st.subheader("Recommended Action")
        st.write(
            "Contact the customer and consider offering a personalized "
            "retention offer, discount, or service upgrade."
        )

    else:
        st.success("LOW CHURN RISK")
        st.write(
            "This customer has a lower likelihood of churning."
        )

        st.subheader("Recommended Action")
        st.write(
            "Continue regular customer engagement and monitor the "
            "customer's future activity."
        )

    st.subheader("Customer Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Tenure**")
        st.write(f"{tenure} months")

    with col2:
        st.write("**Monthly Charges**")
        st.write(f"${monthly_charges:.2f}")

    with col3:
        st.write("**Contract**")
        st.write(contract)
    st.subheader("Prediction History")

    if st.session_state.prediction_history:
        history_df = pd.DataFrame(
        st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.write("No predictions made yet.")

st.divider()

st.subheader("About the Model")

st.write(
    "This application uses a Logistic Regression model trained on "
    "customer demographic, service, contract, and billing information."
)

col1, col2 = st.columns(2)

with col1:
    st.write("**Model:** Logistic Regression")
    st.write("**Prediction Threshold:** 0.40")

with col2:
    st.write("**Training Approach:** Preprocessing + Classification")
    st.write("**Primary Objective:** Identify customers at risk of churn")

st.caption(
    "A lower prediction threshold was selected to improve recall and "
    "identify more potential churn customers."
)