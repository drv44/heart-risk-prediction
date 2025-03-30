import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

# Load dataset (replace with your actual data loading step)
data = pd.read_csv('./dataset/heart.csv')

# Assuming your target variable is 'target' and features are the rest of the columns
X = data.drop(columns='target')
y = data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for models like SVM, Logistic Regression, etc.)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load models
# nn_model = tf.keras.models.load_model('nn.keras')
svm_model = joblib.load('./models/svm.pkl')
rf_model = joblib.load('./models/randomForest.pkl')
logistic_model = joblib.load('./models/logistic.pkl')
knn_model = joblib.load('./models/knn.pkl')

# # Helper function to get probabilities from the neural network model
# def nn_predict_proba(model, X):
#     proba = model.predict(X)
#     return np.hstack([1 - proba, proba])  # Return in a (n_samples, 2) shape for compatibility

# # Wrap the Keras model for use in the VotingClassifier
# class KerasWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, keras_model):
#         self.keras_model = keras_model
        
#     def fit(self, X, y):
#         return self  # The model is already trained
    
#     def predict(self, X):
#         proba = self.keras_model.predict(X)
#         return (proba > 0.5).astype(int).flatten()
    
#     def predict_proba(self, X):
#         return nn_predict_proba(self.keras_model, X)

# # Wrap the neural network model
# nn_model_wrapped = KerasWrapper(nn_model)

# Define the Voting Classifier with all models
voting_clf = VotingClassifier(
    estimators=[
        # ('nn', nn_model_wrapped),
        ('svm', svm_model),
        ('rf', rf_model),
        ('logistic', logistic_model),
        ('knn', knn_model)
    ],
    voting='soft'  # Soft voting to use predicted probabilities
)

# Fit the Voting Classifier on the training data
voting_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = voting_clf.predict(X_test)

# Evaluate accuracy
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))

#save the model
joblib.dump(voting_clf, './models/ensemble.pkl')