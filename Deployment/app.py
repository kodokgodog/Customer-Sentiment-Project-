import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page : ', ('EDA','Predict Sentiment of Hotel Customer Review'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()
