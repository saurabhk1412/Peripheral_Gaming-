import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide")

st.title("Gaming Peripheral Market Intelligence Dashboard")

# Load dataset
df = pd.read_csv("gaming_peripheral_dataset.csv")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    [
        "Overview Dashboard",
        "Descriptive Analytics",
        "Customer Segmentation",
        "Association Insights",
        "Predictive Models",
        "Customer Prediction Tool"
    ]
)

# =============================
# Overview Dashboard
# =============================

if page == "Overview Dashboard":

    st.header("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Respondents", len(df))
    col2.metric("PC Gamers", round((df["Platform"]=="PC").mean()*100,1))
    col3.metric("Mechanical Keyboard Owners", round((df["Mechanical_Keyboard"]==1).mean()*100,1))
    col4.metric("Avg Spending", round(df["Expected_Spending"].mean(),2))

    st.subheader("Platform Distribution")
    fig = px.histogram(df, x="Platform")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("City Tier Distribution")
    fig = px.histogram(df, x="City_Tier")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spending Distribution")
    fig = px.histogram(df, x="Expected_Spending")
    st.plotly_chart(fig, use_container_width=True)


# =============================
# Descriptive Analytics
# =============================

elif page == "Descriptive Analytics":

    st.header("Customer Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(df, x="Age")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Gender Distribution")
        fig = px.histogram(df, x="Gender")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Occupation Distribution")
        fig = px.histogram(df, x="Occupation")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Gamer Identity")
        fig = px.histogram(df, x="Gamer_Identity")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Aesthetic Importance")
        fig = px.histogram(df, x="Aesthetic_Importance")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Expected Spending")
        fig = px.histogram(df, x="Expected_Spending")
        st.plotly_chart(fig, use_container_width=True)


# =============================
# Customer Segmentation
# =============================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    X = df[["Gamer_Identity", "Aesthetic_Importance", "Expected_Spending"]]

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig = px.scatter(
        df,
        x="Expected_Spending",
        y="Gamer_Identity",
        color="Cluster",
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Distribution")
    fig2 = px.histogram(df, x="Cluster")
    st.plotly_chart(fig2, use_container_width=True)


# =============================
# Association Insights
# =============================

elif page == "Association Insights":

    st.header("Association Rule Mining")

    basket = df[
        [
            "Interested_Keycaps",
            "Interested_Cables",
            "Interested_DeskMat",
            "Interested_RGB"
        ]
    ]

    frequent = apriori(basket, min_support=0.1, use_colnames=True)

    rules = association_rules(frequent, metric="confidence", min_threshold=0.3)

    st.subheader("Association Rules Table")
    st.dataframe(rules)

    if len(rules) > 0:

        # Create readable rule column
        rules["rule"] = rules["antecedents"].astype(str) + " → " + rules["consequents"].astype(str)

        st.subheader("Top Rules by Confidence")

        top_conf = rules.sort_values("confidence", ascending=False).head(10)

        fig = px.bar(
            top_conf,
            x="confidence",
            y="rule",
            orientation="h",
            title="Top Association Rules by Confidence"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Rules by Lift")

        top_lift = rules.sort_values("lift", ascending=False).head(10)

        fig2 = px.bar(
            top_lift,
            x="lift",
            y="rule",
            orientation="h",
            title="Top Association Rules by Lift"
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No association rules found. Try adjusting support or confidence thresholds.")


# =============================
# Predictive Models
# =============================

elif page == "Predictive Models":

    st.header("Machine Learning Models")

    features = [
        "Age",
        "Gamer_Identity",
        "Aesthetic_Importance",
        "Mechanical_Keyboard",
        "Expected_Spending"
    ]

    X = df[features]
    y = df["Purchase_Intent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    st.metric("Accuracy", round(accuracy,3))
    st.metric("Precision", round(precision,3))
    st.metric("Recall", round(recall,3))
    st.metric("F1 Score", round(f1,3))

    # Feature importance
    importance = pd.DataFrame({
        "feature":features,
        "importance":clf.feature_importances_
    })

    fig = px.bar(importance, x="feature", y="importance", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)


# =============================
# Customer Prediction Tool
# =============================

elif page == "Customer Prediction Tool":

    st.header("Upload New Customer Data")

    uploaded = st.file_uploader("Upload CSV")

    if uploaded:

        new_data = pd.read_csv(uploaded)

        features = [
            "Age",
            "Gamer_Identity",
            "Aesthetic_Importance",
            "Mechanical_Keyboard",
            "Expected_Spending"
        ]

        X = df[features]
        y = df["Purchase_Intent"]

        model = RandomForestClassifier()
        model.fit(X, y)

        preds = model.predict(new_data[features])

        new_data["Predicted_Interest"] = preds

        st.subheader("Prediction Results")
        st.dataframe(new_data)
