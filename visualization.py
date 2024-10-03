import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


wine = fetch_ucirepo(id=109)
X = pd.DataFrame(wine.data.features, columns=wine.data.feature_names)
y = pd.Series(np.array(wine.data.targets).flatten(), name='class')

print("Назви колонок:", X.columns.tolist())
print("Розмір датасету:", X.shape)

X.fillna(X.mean(), inplace=True)


plt.figure(figsize=(12, 8))
correlation = pd.concat([X, y], axis=1).corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Кореляція ознак')
plt.show()

X.hist(bins=15, figsize=(15, 10))
plt.suptitle('Гістограми розподілу ознак')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='class', y='Alcohol', data=pd.concat([X, y], axis=1))
plt.title('Boxplot Alcohol')
plt.show()
