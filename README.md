# Email Spam Classification with Machine Learning
* url : https://ricky-email-spam-classification.streamlit.app/
This project demonstrates how to build a machine learning model for classifying emails as spam or not spam. The model is trained using natural language processing (NLP) techniques and machine learning algorithms to identify spam emails accurately.

### Libraries Used
* Stopwords: For removing common words that don't add much meaning to the text (e.g., "and", "the").
* Tokenize: To split the text into individual words (tokens).
* String: For removing punctuation and unnecessary characters.
* CountVectorizer: Converts text data into a matrix of token counts.
* TfidfVectorizer: Converts text data into a matrix of Term Frequency-Inverse Document Frequency (TF-IDF) values.
* Scikit-learn: Used for building and evaluating machine learning models.
### Project Steps
#### Data Preprocessing:

* Cleaned and preprocessed email text data by removing stop words, punctuation, and tokenizing the text.
Vectorization:

* Used CountVectorizer and TfidfVectorizer to convert text into numerical features.
Model Training:

* Trained a classification model such as : MultinomialNB, GaussianNB, Xgboost, RandomizeSearchCV, Optuna to predict whether an email is spam or not.
* Focused on optimizing the Recall metric to reduce false negatives (missed spam emails).
### Evaluation:
* Evaluated the model's performance using precision, recall, and accuracy metrics:
* The accuracy is 96% with 100% recall for kaggle datasets.
* According to real world spam, the accuracy is around 70% because the data was trained from my datasets has still not updated, so that some new kind of spam email was not detected.
### Results
The model can effectively classify emails as spam or not, with a focus on minimizing false negatives.
The use of CountVectorizer and TfidfVectorizer significantly improved the model's performance.
  
