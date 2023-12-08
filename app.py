# import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import streamlit as st
import pickle

# load your dataset
data = pd.read_csv("spam.csv", encoding='latin1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'Category', 'v2': 'Text'})

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Text"], data["Category"], test_size=0.2, random_state=42)

# build a pipeline with a CountVectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# save the model as a pickle file
with open("spam_classifier_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Streamlit app
def main():
    st.title("Email Spam or Ham Classifier")

    # user input
    user_input = st.text_input("Enter a message:")

    # make predictions
    if st.button("Predict"):
        prediction = model.predict([user_input])
        st.write(f"Prediction: {prediction[0]}")

    # model testing
    st.header("Model Testing")
    test_accuracy = model.score(X_test, y_test)
    st.write(f"Test Accuracy: {test_accuracy:.2%}")

    y_pred = model.predict(X_test)
    st.subheader("Classification Report:")
    st.text(metrics.classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
