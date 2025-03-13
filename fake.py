import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Load the true and fake news datasets
df_true = pd.read_csv('true.csv')  # Load the real news dataset
df_fake = pd.read_csv('fake.csv')  # Load the fake news dataset

# Step 3: Add label columns
df_true['label'] = 'REAL'  # Assign label 'REAL' to true news
df_fake['label'] = 'FAKE'  # Assign label 'FAKE' to fake news

# Step 4: Combine both datasets
df = pd.concat([df_true, df_fake], ignore_index=True)  # Combine the two dataframes

# Step 5: Prepare the data
X = df['text']  # This is the column with the news text
y = df['label']  # This is the column with 'FAKE' or 'REAL'

# Convert text to numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a basic model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Step 9: Display predictions along with the true labels
print("\nPredictions vs Actual Labels:")
for true_label, predicted_label, text in zip(y_test, predictions, X_test):
    print(f"Text: {text[:100]}...")  # Displaying the first 100 characters of the text
    print(f"True: {true_label}, Predicted: {predicted_label}\n")

# Step 10: Allow user input and predict
user_input = input("Enter a news article text for prediction: ")  # User enters news text

# Transform the user's input into the same format as the training data
user_input_vectorized = vectorizer.transform([user_input])

# Predict whether the entered news is 'REAL' or 'FAKE'
user_prediction = model.predict(user_input_vectorized)

# Output the result
print(f"The entered news article is predicted to be: {user_prediction[0]}")