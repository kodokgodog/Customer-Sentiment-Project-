import streamlit as st
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
nltk.download('stopwords')

import pickle
# Load All Files

with open('t.pickle', 'rb') as file_1:
  t = pickle.load(file_1)

# Setting stopwords with english as default language
stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def text_processing(text):
    # Transforming "n't" into not
    text = text.replace("n't", "not")
    text = text.replace("nt", "not")

    # Converting all text to Lowercase
    text = text.lower()

    # Removing Unicode Characters
    text = re.sub(r'[^\x00-\x7F]', '', text)

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("[^A-Za-z\s']", " ", text)
  
    # Menghilangkan \n
    text = re.sub(r"\\n", " ", text)

    # Removing Whitespace
    text = text.strip()

    # Removing double space
    text = re.sub("\s\s+" , " ", text)
        
    # Tokenizing words
    tokens = word_tokenize(text)

    # Removing Stopwords
    text = ' '.join([word for word in tokens if word not in stopwords])

    # Lemmatizer
    sentence = []
    for word in text.split():
        lem_word = lemmatizer.lemmatize(word, pos='v')  # Lemmatize verb
        lem_word = lemmatizer.lemmatize(lem_word, 'n')  # Lemmatize noun
        sentence.append(lem_word)
    
    text = ' '.join(sentence)

    return text

model_imp = load_model('best_model.h5', compile=False)

def run():
  with st.form(key='Trip_Advisor_Hotel_Review'):
      st.title('Trip Advisor Hotel Review')
      image = Image.open('tripadvisor-logo-6939149F8F-seeklogo.com.png')
      st.image(image)
      st.markdown('---')
      Review = st.text_input('Review',value=' ')

      submitted = st.form_submit_button('Submit')

  df_inf = {
      'Review': Review,

  }

  df_inf = pd.DataFrame([df_inf])
  # Data Inference
  df_inf_copy = df_inf.copy()
  # Applying all preprocessing in one document

  df_inf_copy['Review_processed'] = df_inf_copy['Review'].apply(lambda x: text_processing(x))
  st.dataframe(df_inf_copy)
  # Transform Inference-Set 
  df_inf_transform = df_inf_copy.Review_processed
  df_inf_transform = t.texts_to_sequences(df_inf_transform)
  # Padding the dataset to a maximum review length in words

  df_inf_transform = pad_sequences(df_inf_transform, maxlen=1835)


  if submitted:
      # Predict using Neural Network
      y_pred_inf = model_imp.predict(df_inf_transform)
      y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
      #st.write('# Is the customer at risk of churning ? :thinking_face:')
      if y_pred_inf == 0:
         st.subheader('Negative Feedback')
         st.write('Dear our valued customer,')
         st.write('Thank you for taking the time to share your experience with us. We are sorry to hear that you were not satisfied with your last hotel stay.')
         st.write('We apologize for any inconvenience caused and would like to make it right. Please let us know what specifically went wrong with your stay and we will do everything we can to improve the situation. We value your satisfaction and would appreciate the opportunity to earn back your trust.')
         st.write('Thank you again for bringing this to our attention. We hope to hear from you soon and look forward to the opportunity to serve you better in the future.')
         st.write('Best regards,')
         st.write('Trip Advisor Customer Service Team')
      else:
         st.subheader('Positive Feedback')
         st.write('Dear our valued customer,')
         st.write('Thank you so much for taking the time to provide us with your positive feedback regarding your last hotel reservation/stay.')
         st.write('We always strive to exceed our customers expectations and provide the best possible service. Your satisfaction is our top priority and we are delighted to know that the accomodation can give you a delightful stay.')
         st.write('Best regards,')
         st.write('Trip Advisor Customer Service Team')

if __name__ == '__main__':
    run()