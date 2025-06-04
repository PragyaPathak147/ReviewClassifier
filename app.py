import pandas as pd
import numpy as np 
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



df=pd.read_csv("Combined Data.csv")
df.drop(columns='Unnamed: 0',inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df['statement']=df['statement'].apply(lambda x : x.split())


X1=(df[df['status']=='Normal']['statement']).apply(lambda x : " ".join(x))
X2=(df[df['status']=='Depression']['statement']).apply(lambda x : " ".join(x))
X3=(df[df['status']=='Suicidal']['statement']).apply(lambda x : " ".join(x))
X4=(df[df['status']=='Anxiety']['statement']).apply(lambda x : " ".join(x))
X5=(df[df['status']=='Bipolar']['statement']).apply(lambda x : " ".join(x))
X6=(df[df['status']=='Stress']['statement']).apply(lambda x : " ".join(x))
X7=(df[df['status']=='Personality disorder']['statement']).apply(lambda x : " ".join(x))


Normal=" ".join(X1.iloc[:])
Depression=" ".join(X2.iloc[:])
Suicidal=" ".join(X3.iloc[:])
Anxiety=" ".join(X4.iloc[:])
Bipolar=" ".join(X5.iloc[:])
Stress=" ".join(X6.iloc[:])
Personality=" ".join(X7.iloc[:])


final=pd.DataFrame({'Statements':[Normal,Depression,Suicidal,Anxiety,Bipolar,Stress,Personality],'Status':['Normal','Depression','Suicidal','Anxiety','Bipolar','Stress','Personality']})
cv=CountVectorizer(max_features=1000,stop_words='english')
X=cv.fit_transform(final['Statements']).toarray()

st.title('Review Classifier for mental health')

user_input = st.text_area("Enter your statement here:")
L=[]
L.append(user_input)

# Button to trigger prediction
if st.button("Classify"):
    if user_input:
        V=cv.transform(L).toarray()
        similarity=cosine_similarity(V,X)
        k=np.argmax(similarity)
        st.write(final['Status'][k])
    else:
        st.warning("Please enter a statement before clicking Classify.")


