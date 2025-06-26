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
Feature Extraction: To convert raw text messages into numerical features suitable for machine learning models, TF-IDF (Term Frequency–Inverse Document Frequency) vectorization was applied. The vectorizer was configured to extract the 5,000 most frequent terms across the dataset. The fit_transform() method was used on the training data to learn the vocabulary and compute the TF-IDF scores, while the transform() method was applied to the test data to ensure consistent feature representation based on the learned vocabulary.
Train a Logistic Regression Model using a Logistic Regression classifier was trained using the TF-IDF-transformed training data. This model learns to distinguish between the two sentiment classes—ham (0) and spam (1)—based on the weighted textual features.
Model Evaluation like Accuracy measures the overall correctness of the model's predictions across all samples and classification_report are Precision, Recall, and F1-score provide a more detailed assessment of the model’s performance for each class.
Confusion Matrix: A visual representation of the model’s performance using a confusion matrix highlights the distribution of predictions across four categories: True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP). This visualization helps to clearly identify the types of errors the model is making and provides insights into areas where the model's predictions can be improved.






#output

#Logistic Regression Model accuracy:

![Image](https://github.com/user-attachments/assets/ea93f931-0a48-46bf-909a-43ef1b6a1dde)

#Confusion Matrix Visualization:
![Image](https://github.com/user-attachments/assets/5726d816-1a69-4377-a6de-8f7d2329b76e)

#Model Accuracy Plot:
![Image](https://github.com/user-attachments/assets/809ec0a6-5aba-44d4-8dc8-5085a27e3739)
