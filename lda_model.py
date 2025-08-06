import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder #
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#LDA implementation based on "Implementing linear discriminant analysis (LDA) in Python", a tutorial on the IBM developer website by Eda Kavlakoglu

#to read a .csv
df = pd.read_csv(r"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\dataframes\avg_array.csv")
df = df.drop('EEG epochs', axis=1) #drops the column "EEG epochs"

#divide the dataset into features(x) and target varaible(y)
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

#encode the target variable, this ensures the values are not read as strings
#le = LabelEncoder()
#y = le.fit_transform(y)

#split the data set into training and testing sets
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2, random_state=42)

#LDA
lda = LinearDiscriminantAnalysis(n_components=2)
#fit.transform is used to learn the patterns of the different features and then transforms the data into a lower-dimensions space
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

tmp_df = pd.DataFrame(x_train, columns=['LDA Component 1', 'LDA Component 2'])
tmp_df['Class'] = y_train

sns.FacetGrid(tmp_df, hue ="Class", 
              height=6).map(plt.scatter, 'LDA Component 1', 'LDA Component 2')
plt.legend(loc='upper right')

#RandomForestClassifier is best used for when more than 3 classes are to be classified
classifier = RandomForestClassifier(random_state=42)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_m = confusion_matrix(y_test, y_pred)

#Display the accuracy
print(f'Accuracy: {accuracy:.2f}')

#Display the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
stop=1