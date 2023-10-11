import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_df=pd.read_csv("Girish_Assignment_3.5_Mobile_Price/train.csv")
train_df
print(train_df.isnull().sum())
sns.countplot(x='price_range', data=train_df)
plt.title('Distribution of Price Ranges')
plt.xlabel('Price Range')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,12))
sns.heatmap(train_df.corr() , annot=True , cmap="inferno");
sns.boxplot(x='price_range', y='battery_power', data=train_df)
plt.title('Battery Power vs. Price Range')
plt.xlabel('Price Range')
plt.ylabel('Battery Power')
plt.show()
# Define features (X) and the target variable (y)
X = train_df.drop("price_range", axis=1)
y = train_df["price_range"]
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

models = [
    ('LogisticRegression', LogisticRegression()),
    ('SVM', SVC()), 
    ('RandomForest', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('DecisionTree', DecisionTreeClassifier()),
    ('NaiveBayes', GaussianNB())
]
for name, model in models:
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print accuracy
    print(f'{name} test accuracy: {accuracy:.3f}')
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)
from sklearn.linear_model import LogisticRegression
# Create a LogisticRegression model
logistic_reg = LogisticRegression()
# Fit the model to the training data
logistic_reg.fit(X_train, y_train)
# Make predictions on the test data
logistic_reg_predictions =logistic_reg.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, logistic_reg_predictions)
print("Accuracy of Logistic Regression:", accuracy)
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_rep)
prediction = logistic_reg.predict(X_test)
test_df=pd.read_csv("Girish_Assignment_3.5_Mobile_Price/test.csv")
test_df
test_df = test_df.drop(['id'], axis = 1)
test_df.shape
X_test_scaled = scalar.transform(test_df)
testPrediction= logistic_reg.predict(X_test_scaled)
test_df['predicted_price'] = testPrediction
test_df
