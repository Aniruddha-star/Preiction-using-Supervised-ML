
""" First we import some important libraries which we would be needing
    to complete this task """

import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt

""" Now we import our data from the url provided """

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)

""" Now we enter distribution scores and plot them according to 
   the requirement """

s_data.plot(x='Hours', y='Scores', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Score')    
plt.show()

""" Then we divide the data into attributes and labels """

X = s_data.iloc[:, :-1].values    
y = s_data.iloc[:, 1].values

""" The split of data into the training and test sets is very important 
    as in this time we will be using Scikit Learn's builtin method of 
    train_test_split(), as below: """

from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

""" The very next process is to train the algorithm """

from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
  
print("Training ... Completed !.")

""" Now implement the plotting test data using the previously trained 
    test data """

line = regressor.coef_*X+regressor.intercept_  
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show()  

""" Predicting the scores for the model is the next important step 
    towards knowing our model """

print(X_test)   
y_pred = regressor.predict(X_test)  

""" Then we compare the actual versus predicted model to understand 
    our model fitting """

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df

""" What wiil be the predicted score if the student studies 
    for 9.25hr/day? """

hours = [[9.25]]  
own_pred = regressor.predict(hours)  
print("Number of hours the student studied = {}".format(hours))  
print("Predicted score that the student would get = {}".format(own_pred[0]))