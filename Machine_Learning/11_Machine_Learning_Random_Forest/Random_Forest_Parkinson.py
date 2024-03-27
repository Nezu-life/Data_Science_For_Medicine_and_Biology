# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Read CSV file into DataFrame
fileName = "../../datasets/parkinsons.csv"
df = pd.read_csv(fileName)

# Average variables by 'name', ignoring 'test', 'name'
df_grouped = df.drop(columns=['test', 'name']).groupby(df['name']).mean()

# Prepare the features and target variable
X = df_grouped.drop(columns=['status']).values
y = df_grouped['status'].values

# Number of splits and repetitions
n_splits = 3
n_repeats = 20

# Initialize variables to store results
fold_results = {i: [] for i in range(n_repeats)}

# Define the stratified K-fold split
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Train and test the Random Forest classifier
for fold, (train_index, test_index) in enumerate(rskf.split(X, y)):
    print(f"Fold {fold}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate balanced accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    fold_results[fold % n_repeats].append(bal_acc)

# Plotting the barplot with standard deviations
fold_means = [np.mean(fold_results[i]) for i in range(n_repeats)]
fold_stds = [np.std(fold_results[i]) for i in range(n_repeats)]

print(f"Average accuracy of all folds: {np.mean(fold_means)}:")

sns.barplot(x=list(range(n_repeats)), y=fold_means, yerr=fold_stds[0], hue=list(range(n_repeats)), legend=False, palette='rainbow')
plt.xlabel('Fold')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy with Standard Deviation for each fold')
plt.show()
