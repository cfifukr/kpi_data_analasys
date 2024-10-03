import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# https://archive.ics.uci.edu/dataset/109/wine
wine = fetch_ucirepo(id=109)
X = pd.DataFrame(wine.data.features, columns=wine.data.feature_names)
y = wine.data.targets


print(X.isnull().sum())

X_cleaned = X.dropna()
y_cleaned = y[:len(X_cleaned)]
print(X_cleaned.shape)

# візуалізація в visualization.py


scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_cleaned)
X_normalized_df = pd.DataFrame(X_normalized, columns=X_cleaned.columns)
print(X_normalized_df.head())


X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y_cleaned, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
