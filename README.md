
# Reservation Cancelation Prediction

Project ini merupakan project mandiri. Project ini memiliki tujuan untuk memanfaatkan Machine Learning guna memprediksi klasisifikasi dari tamu hotel yang mereservasi kamar apakah ia akan mencancel reservasinya atau tidak. Sehingga dapat meminimalisir pengaruh dari adanya cancelation tersebut.

## Deployment Link
Deployment : (https://huggingface.co/spaces/kodokgodog/Sentiment_Prediction_of_Trip_Advisor_Hotel_Reviews)

## Permasalahan

Permasalahan yang ada di dalam project ini adalah, menganalisis dan mengembangkan NLP yang kuat dan akurat yang dinyatakan oleh ulasan pelanggan di Trip Advisor Hotel Reviews dan membedakannya apakah sentimen tersebut negatif atau positif. Hal ini membantu untuk memahami pendapat dan kepuasan keseluruhan pelanggan. Tujuan utamanya adalah mencapai akurasi tinggi dalam klasifikasi sentimen.

## Deskripsi Data

Dataset ini terdiri dari informasi review dari para pengguna hotel terhadap hotel-hotel yang telah mereka gunakan melalui aplikasi Trip Advisor.

## Evaluasi Model

Model akan dievaluasi berdasarkan kemampuannya dalam mengklasifikasikan sentimen dari review pengguna hotel baik itu sentimen negatif atau positif. NLP yang dibuat menggunakan LSTM dan GRU serta kedua algoritma tersebut yang telah di improve. Metrik evaluasi utamanya adalah akurasi, yang memberikan penilaian terhadap kemampuan model dalam membedakan antara sentimen negatif dan positif.

## Kesimpulan

EDA Conclusion:

The distribution of data is imbalance, where the positive ratings is having a lot more data compared to the negative ratings. Eventhough the positive one is higher, but the negative review is really important to be reviewed too.
The key difference between positive and negative reviews is the attitude of the reviewer and their overall satisfaction with the hotel service where they stay. Positive reviews reflect a positive experience, while negative reviews reflect a negative experience.
Modelling Conclusion:

Based on the Model Analysis conducted, it is decided that the improved LSTM model is the best NLP modeling approach to predict sentiment of Trip Advisor Hotel review. The following findings are obtained regarding the model used:

With evaluation score of all accuracy 83% the models is good enough on the score, but if the model gonna be used on business it still need improvement so at least it can go up to 90% accuracy before it can be used.
Improved LSTM has slightly better precision for class 0, while Improved GRU has slightly better precision for class 1. However, Improved LSTM has slightly better recall, f1-score, and accuracy overall.
Business Insight:

From all of the review we can see the Sentiment analysis on Trip Advisor Hotel reviews can provide valuable insights for businesses, including identifying common issues that customers face with their products or services, understanding the factors that drive customer satisfaction, and tracking changes in customer sentiment over time.
For the modelling that has been made the modeling process can indeed be further improved, particularly in terms of hyperparameter tuning. Searching more optimal paramaters that can be used, potentially leading to better performance of the model.

Hyperparameter tuning is a crucial step in optimizing the model's performance. By systematically varying hyperparameters and evaluating the model's performance on validation data.

If more time can be put into hyperparameter tunings, its not impossible to get more accurate models that can predict Customer Churn better.

We can improve this model from the data preprocessing too, such as reducing the vocabulary on the data, because from the re-scanning the data, there are words that still need preprocessing like words that appear in both reviews (adding words that appeared in both reviews into the stopwords), typo on the word on the review, broken word that doesn't mean anything, etc.
