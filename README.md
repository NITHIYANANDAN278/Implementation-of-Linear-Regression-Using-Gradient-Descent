# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import numpy as np.

2.Give the header to the data.

3.Find the profit of population.

4.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:NITHIYANANDAN N
RegisterNumber:212222230099
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate error
        errors=(predictions - y).reshape(-1,1)
        
        theta -= learning_rate*(1/len(X1)*X).T.dot(errors)
        
    return theta
data=pd.read_csv("C:/Users/SEC/Downloads/50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```


## Output:
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121784636/5d5b49a2-a99f-4713-a098-f705757dd96d)
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121784636/a4606269-b78a-4568-a67d-9ab892980b45)
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121784636/a55925d2-cb9a-461b-8d77-b801746f281f)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
