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
The tokenizer transforms words into integer representations (e.g., "the" → 1, "movie" → 2), while restricting the vocabulary to the top 5,000 most frequent words. Words not encountered during training are represented by a special <OOV> (Out Of Vocabulary) token.Converts each review/sentence into a list of word indices based on the tokenizer's vocabulary.while Padding was ensures all sequences have a uniform length of 100 (maxlen=100).Shorter sequences are padded with zeros, while longer ones are truncated.Using padding='post' means padding is added to the end of each sequence.
Converts labels to the appropriate format ('int64' i.e.. integer) required for training with a Keras model.
Buliding the LSTM model converts word indices into 64-dimensional dense vectors (input_dim=5000, input_length=100), followed by a recurrent layer (LSTM) to learn text sequences. A 50% dropout prevents overfitting. The final sigmoid-activated neuron outputs a probability (0–1) for class 1 ("positive"). With a threshold of 0.5, predictions are classified as positive or negative. The model uses binary_crossentropy loss (for binary classification), Adam optimizer (with adaptive learning rate), and tracks accuracy as the evaluation metric.LSTM model is trained using the fit() function for 5 epochs on the padded training data and corresponding labels. During each epoch, the model's performance is evaluated on the validation set to monitor generalization. The training process displays real-time updates, including loss and accuracy metrics, and returns a history object that stores the performance trends across all epochs.
The trained LSTM model is evaluated on the test dataset using the evaluate() function, which calculates the final loss and accuracy. This step assesses how well the model generalizes to unseen data. The resulting test accuracy is printed, providing a quantitative measure of the model’s predictive performance on the test set.
Plotting Model Accuracy Over Training a line plot is generated to visualize the LSTM model's accuracy over each training epoch, showing both training and validation accuracy curves. The training accuracy reflects how well the model learns from the training data, while the validation accuracy indicates its ability to generalize to unseen data. This plot is a useful diagnostic tool to assess the model's learning behavior—helping identify whether it is learning effectively, overfitting (performing well on training data but poorly on validation), or underfitting (performing poorly on both).Finally a model accuracy plot is generated.
This sentiment analysis uses key NLP techniques such as text preprocessing, tokenization, and padding to prepare the data. Word embeddings capture semantic meaning, while an LSTM layer learns sequential patterns. Dropout helps prevent overfitting, and a sigmoid-activated output with binary crossentropy loss is used for final sentiment prediction.


#output

#Logistic Regression Model accuracy:

![Image](https://github.com/user-attachments/assets/ea93f931-0a48-46bf-909a-43ef1b6a1dde)

#Confusion Matrix Visualization:
![Image](https://github.com/user-attachments/assets/5726d816-1a69-4377-a6de-8f7d2329b76e)

#Model Evaluation on test data and reports the accuracy:
![Image](https://github.com/user-attachments/assets/a6373a50-a444-40f5-b3c7-8755ce7dfc05)

#Model Accuracy Plot:
![Image](https://github.com/user-attachments/assets/809ec0a6-5aba-44d4-8dc8-5085a27e3739)
