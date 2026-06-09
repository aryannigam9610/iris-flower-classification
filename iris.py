import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Iris.csv")

print("Dataset Preview:")
print(df.head())

feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_cols]

y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy =", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample_data = [[5.1, 3.5, 1.4, 0.2]]
sample_df = pd.DataFrame(sample_data, columns=feature_cols)

prediction = model.predict(sample_df)

print("\nPredicted Species:", prediction[0])
