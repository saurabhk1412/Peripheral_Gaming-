
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide")
st.title("Gaming Peripheral Market Intelligence Platform")

uploaded = st.file_uploader("Upload dataset (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("gaming_peripheral_dataset.csv")

page = st.sidebar.selectbox("Navigation",[
"Overview",
"Descriptive Analytics",
"Customer Segmentation",
"Association Insights",
"Predictive Models",
"Customer Prediction Tool"
])

if page=="Overview":
    st.subheader("Executive Overview")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Respondents",len(df))
    col2.metric("PC Gamers %",(df["Platform"].value_counts(normalize=True).get("PC",0)*100).round(1))
    col3.metric("Mechanical Keyboard %",(df["Mechanical_Keyboard"].value_counts(normalize=True).get("Yes",0)*100).round(1))
    col4.metric("Avg Spending",round(df["Expected_Spending"].mean(),0))

    c1,c2 = st.columns(2)
    fig=px.histogram(df,x="Platform",title="Gaming Platform Distribution")
    c1.plotly_chart(fig,use_container_width=True)
    fig=px.histogram(df,x="Purchase_Interest",title="Purchase Interest Distribution")
    c2.plotly_chart(fig,use_container_width=True)

    c3,c4 = st.columns(2)
    fig=px.histogram(df,x="City_Tier",title="City Tier Distribution")
    c3.plotly_chart(fig,use_container_width=True)
    fig=px.histogram(df,x="Expected_Spending",title="Spending Distribution")
    c4.plotly_chart(fig,use_container_width=True)

elif page=="Descriptive Analytics":
    st.subheader("Demographic Insights")
    col1,col2,col3 = st.columns(3)
    col1.plotly_chart(px.histogram(df,x="Age_Group",title="Age Distribution"),use_container_width=True)
    col2.plotly_chart(px.histogram(df,x="Gender",title="Gender Distribution"),use_container_width=True)
    col3.plotly_chart(px.histogram(df,x="Occupation",title="Occupation Distribution"),use_container_width=True)

    st.subheader("Gaming Behaviour")
    col1,col2,col3 = st.columns(3)
    col1.plotly_chart(px.histogram(df,x="Platform",title="Platform"),use_container_width=True)
    col2.plotly_chart(px.histogram(df,x="Gamer_Identity",title="Gamer Identity Level"),use_container_width=True)
    col3.plotly_chart(px.histogram(df,x="Mechanical_Keyboard",title="Mechanical Keyboard Ownership"),use_container_width=True)

    st.subheader("Setup & Aesthetic Insights")
    col1,col2,col3 = st.columns(3)
    col1.plotly_chart(px.histogram(df,x="Preferred_Aesthetic",title="Aesthetic Preference"),use_container_width=True)
    col2.plotly_chart(px.histogram(df,x="Setup_Value",title="Setup Value"),use_container_width=True)
    col3.plotly_chart(px.scatter(df,x="Aesthetic_Importance",y="Expected_Spending",title="Aesthetic Importance vs Spending"),use_container_width=True)

    st.subheader("Product Interest")
    interest=df[["Interested_Keycaps","Interested_Cables","Interested_DeskMat","Interested_RGB"]].sum()
    st.plotly_chart(px.bar(interest,title="Accessory Interest Counts"),use_container_width=True)

elif page=="Customer Segmentation":
    st.subheader("Customer Clustering")
    features=df[["Gamer_Identity","Aesthetic_Importance","Expected_Spending"]]
    kmeans=KMeans(n_clusters=4,random_state=42)
    df["Cluster"]=kmeans.fit_predict(features)

    st.plotly_chart(px.scatter(df,x="Expected_Spending",y="Gamer_Identity",color=df["Cluster"].astype(str),
    title="Customer Segments"),use_container_width=True)

    st.plotly_chart(px.histogram(df,x="Cluster",title="Cluster Distribution"),use_container_width=True)

elif page=="Association Insights":
    st.subheader("Product Association Rules")
    basket=df[["Interested_Keycaps","Interested_Cables","Interested_DeskMat","Interested_RGB"]]
    freq=apriori(basket,min_support=0.1,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=0.3)

    if len(rules)>0:
        st.dataframe(rules[["antecedents","consequents","confidence","lift"]])
        st.plotly_chart(px.bar(rules.sort_values("confidence",ascending=False).head(10),
        x="confidence",y=rules.index.astype(str),title="Top Rules by Confidence"),use_container_width=True)
        st.plotly_chart(px.bar(rules.sort_values("lift",ascending=False).head(10),
        x="lift",y=rules.index.astype(str),title="Top Rules by Lift"),use_container_width=True)
    else:
        st.write("No strong association rules found.")

elif page=="Predictive Models":
    st.subheader("Classification Model")

    target=df["Purchase_Interest"]
    features=df.drop(columns=["Purchase_Interest"])
    X=pd.get_dummies(features)
    le=LabelEncoder()
    y=le.fit_transform(target)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)

    acc=accuracy_score(y_test,pred)
    prec=precision_score(y_test,pred,average="weighted")
    rec=recall_score(y_test,pred,average="weighted")
    f1=f1_score(y_test,pred,average="weighted")

    st.write("Accuracy",acc)
    st.write("Precision",prec)
    st.write("Recall",rec)
    st.write("F1 Score",f1)

    importances=clf.feature_importances_
    feat=pd.Series(importances,index=X.columns).sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(feat,title="Feature Importance"),use_container_width=True)

    st.subheader("Regression Model")

    reg_target=df["Expected_Spending"]
    X_train,X_test,y_train,y_test=train_test_split(X,reg_target,test_size=0.2,random_state=42)
    reg=RandomForestRegressor(n_estimators=200)
    reg.fit(X_train,y_train)
    pred_spend=reg.predict(X_test)

    st.plotly_chart(px.scatter(x=y_test,y=pred_spend,labels={"x":"Actual","y":"Predicted"},
    title="Actual vs Predicted Spending"),use_container_width=True)

elif page=="Customer Prediction Tool":
    st.subheader("Predict New Customers")

    new_file=st.file_uploader("Upload new customer dataset",type=["csv"])
    if new_file:
        new_df=pd.read_csv(new_file)

        target=df["Purchase_Interest"]
        features=df.drop(columns=["Purchase_Interest"])
        X=pd.get_dummies(features)
        le=LabelEncoder()
        y=le.fit_transform(target)

        clf=RandomForestClassifier(n_estimators=200)
        clf.fit(X,y)

        reg=RandomForestRegressor(n_estimators=200)
        reg.fit(X,df["Expected_Spending"])

        new_encoded=pd.get_dummies(new_df)
        new_encoded=new_encoded.reindex(columns=X.columns,fill_value=0)

        pred_interest=clf.predict(new_encoded)
        pred_spending=reg.predict(new_encoded)

        result=new_df.copy()
        result["Predicted_Interest"]=le.inverse_transform(pred_interest)
        result["Predicted_Spending"]=pred_spending
        st.dataframe(result)
