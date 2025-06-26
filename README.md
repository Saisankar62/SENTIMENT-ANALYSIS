# SENTIMENT-ANALYSIS
**COMPANY**: CODTECH IT SOLUTIONS
**NAME**: BALLEDA SAISANKAR
**INTERN ID**: CT08DF1488
**DOMAIN**: DATA ANALYTICS
**DURATION**: 8 WEEKS
**MENTOR**: NEELA SANTOSH


#Description of the task
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
I sucessfully completed the Sentiment Analysis on IMDB reviews dataset. Library Imported like pandas, numpy-->For data handling,nltk -->For text cleaning (stopwords, tokenization), sklearn-->For ML modeling and evaluation, tensorflow.keras -->For deep learning models like LSTM, matplotlib/seaborn -->For visualization data.
Dataset used like SMS spam/ham dataset from GitHub as the primary data source,the dataset was loaded directly into the environment using standard data handling libraries.
Label Mapping -->Original labels are 'ham' → good text, 'spam' → bad text.Then we mapped to 'positive' and 'negative' next converted to numeric: {'positive': 1, 'negative': 0}.
Text cleaning was done usng a function clean_text() was applied to Remove special characters,Normalize whitespace,Convert to lowercase,Remove stopwords,Resulting in a new column to 'cleaned_text'.
The final cleaned dataset was divided into training and testing sets to evaluate model performance effectively. The text inputs were split into X_train and X_test, while the corresponding binary sentiment labels (0 for non-spam/ham and 1 for spam) were split into y_train and y_test. 

#output

#Logistic Regression Model accuracy:

![Image](https://github.com/user-attachments/assets/ea93f931-0a48-46bf-909a-43ef1b6a1dde)

#Confusion Matrix Visualization:
![Image](https://github.com/user-attachments/assets/5726d816-1a69-4377-a6de-8f7d2329b76e)

#Model Accuracy Plot:
![Image](https://github.com/user-attachments/assets/809ec0a6-5aba-44d4-8dc8-5085a27e3739)
