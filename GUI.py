import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
from importlib import reload
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

header=st.container()
dataset=st.container()
features=st.container()
modelTraining=st.container()

def cluster_function(prediction):
    if prediction =="Cluster 0":
        return "Regular"
    elif prediction =="Cluster 1":
        return "Loyal"
    elif prediction =="Cluster 2":
        return "Star"
    elif prediction =="Cluster 3":
        return "Big Spender"
    return "Lost Cheap"

## Function to check skewness
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return




with header:
    st.title("Say hi!!!")
    st.text("My project is .......")
    
with dataset:
    st.title("E-commerce Data Table")
    
    data_RFM = pd.read_csv("Data/RFM_data.csv")
    st.write(data_RFM.head())
    
    
    st.subheader("Top 10 customer with high frequency of spending")
    
    most_customer=data_RFM.groupby('customer_id').size().reset_index()
    most_customer.columns = ['customer_id', 'frequency']
    most_customer.sort_values("frequency",ascending=False)
    most_customer_10=most_customer.sort_values("frequency",ascending=False).head(10)
    

    fig = plt.figure(figsize=(11, 7))
    sns.barplot(x="customer_id", y="frequency", data=most_customer_10, palette="Blues_d")
    st.pyplot(fig)
    
    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_agg = data_RFM.groupby('RFM_Level').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

    # Reset the index
    rfm_agg = rfm_agg.reset_index()
    
    st.write(rfm_agg.head())
    
    fig2=plt.figure(figsize=(8, 6))
    plt.pie(data_RFM.RFM_Level.value_counts(),labels=data_RFM.RFM_Level.value_counts().index,autopct='%.0f%%')
    plt.title("Percentage of each customer segment")
    st.pyplot(fig2)
    
    #Create our plot and resize it.
    fig3 = plt.gcf()
    fig3.set_size_inches(14, 10)

    colors_dict = {'ACTIVE':'yellow','BIG SPENDER':'royalblue', 'LIGHT':'cyan',
                'LOST':'red', 'LOYAL':'purple', 'POTENTIAL':'green', 'STARS':'gold'}

    squarify.plot(sizes=rfm_agg['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                        for i in range(0, len(rfm_agg))], alpha=0.5 )


    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    st.pyplot(fig3)

with features:
    st.header("The features I created")
    
    st.markdown("* **first feature: ** I created this feature because of this...I calculated it using this logic...")
    st.markdown("* **second feature: ** I created this feature because of this...I calculated it using this logic...")
    
with modelTraining:
    st.header("Time to Train the Model!")
    st.text("Here you get to choose the hyperparameters of the model and see how the perfomance change")
    
    sel_col, disp_col=st.columns(2)
    
    max_depth=sel_col.slider("What should be the max_depth of the model?",min_value=1,max_value=20,value=2,step=1)
    
    #n_estimators=sel_col.selectbox("How many should there be?",options=[100,200,300,"No limit"],index=0)
    
    #input_feature=sel_col.text_input("Which feature should be used as the input feature?","PULoad")
    
    df_now = data_RFM[['Recency','Frequency','Monetary']]
    
    RFM_Table_scaled=df_now.copy()
    
    
    RFM_Table_scaled = np.log(RFM_Table_scaled+1)

    # plt.figure(figsize=(9, 9))
    # plt.subplot(3, 1, 1)
    # check_skew(RFM_Table_scaled,'Recency')
    # plt.subplot(3, 1, 2)
    # check_skew(RFM_Table_scaled,'Frequency')
    # plt.subplot(3, 1, 3)
    # check_skew(RFM_Table_scaled,'Monetary')
    # plt.tight_layout()
    
    
    scaler = StandardScaler()
    scaler.fit(RFM_Table_scaled)
    RFM_Table_scaled = scaler.transform(RFM_Table_scaled)
    

    
    model = KMeans(n_clusters=max_depth, random_state=42)
    model.fit(RFM_Table_scaled)
    
    df_now["Cluster"] = model.labels_
    df_now.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':['mean', 'count']}).round(2)
    
    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_agg_kmeans_k_5 = df_now.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg_kmeans_k_5.columns = rfm_agg_kmeans_k_5.columns.droplevel()
    rfm_agg_kmeans_k_5.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg_kmeans_k_5['Percent'] = round((rfm_agg_kmeans_k_5['Count']/rfm_agg_kmeans_k_5.Count.sum())*100, 2)

    # Reset the index
    rfm_agg_kmeans_k_5 = rfm_agg_kmeans_k_5.reset_index()

    # Change thr Cluster Columns Datatype into discrete values
    rfm_agg_kmeans_k_5['Cluster'] = 'Cluster '+ rfm_agg_kmeans_k_5['Cluster'].astype('str')

    rfm_agg_kmeans_k_5["type"]=rfm_agg_kmeans_k_5["Cluster"].map(lambda x: (cluster_function(x)))
    
    st.write(rfm_agg_kmeans_k_5)
    
    

    
    # st.write(data_RFM.head())
    
    
    