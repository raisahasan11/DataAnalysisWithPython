
#import Library

import pandas as pd
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("lastfile3.csv")
#print(data)

#Data Preprocessing
dataFill=data.fillna(value=500)
pd.set_option('display.max_column',None)
# print(dataFill)

#Slicing the dataset
x=dataFill.iloc[:,7:8].values
y=dataFill.iloc[:,9].values
# print(x)

# splitting the dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# print(x_train)
# print(x_test)

from sklearn.preprocessing import MinMaxScaler
scale1=MinMaxScaler()
x_train=scale1.fit_transform(x_train)
# print(x_train)
# print(x_test)


from sklearn.svm import SVC
cli=SVC(kernel="linear",random_state=0)
cli.fit(x_train,y_train)
y_predict=cli.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_predict)
print(cm)
acc=accuracy_score(y_test,y_predict)
print(acc)

# graph

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(data["Income per month"], data["Member "], color='r', alpha=0.5, edgecolors='k')
plt.title("Income per month vs Members ")
plt.grid(ls='--', alpha=0.7)
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
data["Income per month"].plot(kind='hist', bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.title("Distribution of Income per month")
plt.grid(ls='--', alpha=0.7)
plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
data.boxplot(column=["Income per month"], patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title("Income per month Box Plot")
plt.grid(ls='--', alpha=0.7)
plt.show()

# Bar Plot
plt.figure(figsize=(8, 6))
data.plot(kind="bar", x="Income per month", y="Member ", color='purple', alpha=0.7)
plt.title("Income per month vs Members ")
plt.grid(ls='--', alpha=0.7)
plt.show()

# Pie Chart
plt.figure(figsize=(8, 6))
data.groupby("Income per month")["Member "].sum().plot(kind='pie', autopct='%1.1f%%', colormap='viridis')
plt.ylabel("")
plt.title("Members  Distribution by Income per month")
plt.show()