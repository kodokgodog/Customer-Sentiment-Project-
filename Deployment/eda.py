import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
import wordcloud
from wordcloud import WordCloud

from PIL import Image

st.set_page_config(
    page_title='Sentiment Analysis of Trip Advisor Hotel Reviews',
    initial_sidebar_state='expanded'
)

def run():
    # title
    st.title('Sentiment Analysis of Trip Advisor Hotel Reviews')
    st.write('by Satriya Fauzan Adhim')

    # sub header
    st.subheader ('Exploratory Data Analysis of the Dataset.')

    # Add Image
    image = Image.open('images.png')
    st.image(image,caption = 'Trip Advisor')

    # Description
    st.write('''Tripadvisor, the world's largest travel guidance platform, helps hundreds of 
    millions of people each month** become better travelers, from planning to booking to taking a trip. 
    Travelers across the globe use the Tripadvisor site and app to discover where to stay, what to do and
    where to eat based on guidance from those who have been there before. With more than 1 billion reviews 
    and opinions of nearly 8 million businesses, travelers turn to Tripadvisor to find deals on 
    accommodations, book experiences, reserve tables at delicious restaurants and discover great places 
    nearby. Sentiment analysis on Trip Advisor Hotel reviews can **provide valuable insights for businesses**, 
    including identifying common issues that customers face with their products or services, 
    understanding the factors that drive customer satisfaction, and tracking changes in customer sentiment 
    over time.''')
    st.write('# Dataset') 
    st.write(''' Dataset used is Trip Advisor Hotel Reviews dataset from [kaggle]
    ("https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews").''')

    # show dataframe
    df2 = pd.read_csv('tripadvisor_hotel_reviews.csv')
    st.dataframe(df2)
    # add description of Dataset
    st.write('''In this dataset, Rating 1,2 and 3 is the **negative review** and Rating 4 and 5 is the **positive review**''')

    ###
    # create a copy of the dataframe
    df_eda = df2.copy()
    df_eda.Rating.replace({1:'Negative Review',2:'Negative Review',3:'Negative Review',4:'Positive Review',5:'Positive Review'}, inplace=True)
    # Separating positive & negative review
    positive_review = df_eda[df_eda['Rating']=='Positive Review']
    negative_review = df_eda[df_eda['Rating']=='Negative Review']
    
    st.write('#### Plot Rating')
    fig = plt.figure(figsize=(15, 5))
    sns.countplot(x='Rating', data=df_eda)
    st.pyplot(fig)

    st.write(
    '''
    From the visualization above, we can see that the amount of customer that giving positive review
    is higher than the negative one. But there are still some of negative review that customer give.
    ''')

    st.markdown('---')

    st.write('#### Word Cloud')
    text_cloud = ' '.join(df_eda['Review'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_cloud)
    fig, ax = plt.subplots(figsize=(10, 5))  # Buat objek figure dan axes pada matplotlib
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()

    # Tampilkan word cloud menggunakan st.pyplot() dengan menyertakan argumen figure
    st.pyplot(fig)

    st.write(
    '''
    From the word cloud above we can know about the some of the word that are frequently used on the review.
    From there we can see the aspect that are getting mentioned a lot on the review like room, 
    restaurant, breakfast etc. From the word cloud we can see too that a lot of positive word like 
    good and nice. But from the word cloud alone we can't really see a lot of information and its 
    just for a rough representation about the review data, so we need more detailed analysis about 
    the data.
    ''')



    st.markdown('---')

    st.write('#### Sample of Review')
    # Print sample reviews
    pd.set_option('display.width', None)
    sample_negative_review = df_eda[df_eda['Rating'] == 'Negative Review'].sample(3)
    sample_positive_review = df_eda[df_eda['Rating'] == 'Positive Review'].sample(3)

    # Print Sample of Negative Review
    st.write('Example of Negative Reviews')
    st.write('-' * 100)
    for review in sample_negative_review['Review']:
        st.write(review)
    st.write('-' * 100)

    # Print Sample of Positive Review
    st.write('Example of Positive Reviews')
    st.write('-' * 100)
    for review in sample_positive_review['Review']:
        st.write(review)
    st.write('-' * 100)


if __name__ == '__main__':
    run()