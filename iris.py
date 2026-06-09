import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("Iris.csv")

# First 5 rows dekhne ke liye
print("Dataset Preview:")
print(df.head())

# Features (Input)
# नोट: अगर आपके CSV में 'Id' नाम का कॉलम है, तो उसे यहाँ छोड़ना बिल्कुल सही फैसला था।
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_cols]

# Target (Output)
y = df['Species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy =", accuracy * 100, "%")

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- यहाँ बदलाव किया गया है ताकि Warning न आए ---
# टेस्ट के लिए नए फूल का डेटा DataFrame के रूप में दे रहे हैं, कॉलम नामों के साथ
sample_data = [[5.1, 3.5, 1.4, 0.2]]
sample_df = pd.DataFrame(sample_data, columns=feature_cols)

# Prediction on new sample
prediction = model.predict(sample_df)

print("\nPredicted Species:", prediction[0])