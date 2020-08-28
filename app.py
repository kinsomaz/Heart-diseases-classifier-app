import streamlit as st
import numpy as np
import pandas as pd
from PIL import image
from sklearn.neighbors import KNeighborsClassifier


def welcome():
return "Welcome All"

def predict_heart_diseases( age , sex ,cp ,trestbps , chol , fbs , restecg , thalach , exang , oldpeak , slope , ca , thal):
  classifier = KNeighborsClassifier(n_neighbors = 7)
  classifier.fit(x_train.T,y_train.T)
  prediction = classifier.predict([[age , sex ,cp ,trestbps , chol , fbs , restecg , thalach , exang , oldpeak , slope , ca , thal]])
  print(prediction)
  return prediction

def main():
  st.title("Heart-Diseases Classifier")
  html_temp = """
  <div style="backgroud-color:tomato ; padding:10px">
  <h2 style="color:white ; text-align: center ;">Streamlit Heart-Diseases Classifier ML app </h2> 
  st.markdown(html_temp , unsafe_allow_html = True )
  
  @st.cache(persist=True)
  def load_data():
    data = pd.read_csv("heart.csv")
    return data
  
  @st.cache(persist=True)
  def split(df):
    y = df.target.values
    x_data = df.drop(['target'], axis = 1)
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
    return x_train, x_test, y_train, y_test
    
  
  df = load_data()
  
  
  x_train, x_test, y_train, y_test = split(df)
  x_train = x_train.T
  y_train = y_train.T
  x_test = x_test.T
  y_test = y_test.T
    
    
    
  age = st.text_input("age", "Type Here")
  sex = st.text_input("age", "Type Here")
  cp = st.text_input("age", "Type Here")
  trestbps = st.text_input("age", "Type Here")
  chol = st.text_input("age", "Type Here")
  fbs = st.text_input("age", "Type Here")
  restecg = st.text_input("age", "Type Here")
  thalach = st.text_input("age", "Type Here")
  exang = st.text_input("age", "Type Here")
  oldpeak = st.text_input("age", "Type Here")
  slope = st.text_input("age", "Type Here")
  ca = st.text_input("age", "Type Here")
  thal = st.text_input("age", "Type Here")
  results = ""
  if st.button("Predict"):
    result = predict_heart_diseases( age , sex ,cp ,trestbps , chol , fbs , restecg , thalach , exang , oldpeak , slope , ca , thal)
  st.success('The output is {} '.format(result))
  if st.button("About"):
    st.text("Lets LEarn")
    st.text("Built with Streamlit")

if __name == '__main__':
main()
