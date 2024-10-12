import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_folder = "output_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


data = pd.read_csv("./data/student_scores.csv")
print(' ')
print('======= Head(5) ========')
print(data.head())


sns.pairplot(data, x_vars=['Hours'], y_vars='Scores', height=4, aspect=1.5) 
plt.savefig(f"{output_folder}/scatterplot.png")
plt.show()

# create a features (X) and targets (y)
X = data[['Hours']]
y = data[['Scores']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# instantiate the model
linreg = LinearRegression()

linreg.get_params()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

print(' ')
print('=========== the intercept and coefficients ============')
print("Y-intercept: ",linreg.intercept_)
print("Coefficient: ",linreg.coef_)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

print(' ')
print('======= MAE using scikit-learn ========')
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


y_pred = linreg.predict(X)


plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Best fit line')


plt.xlabel('Hours of Study')
plt.ylabel('Predicted Score')
plt.title('Linear Regression Best Fit Line')
plt.legend()
plt.savefig(f"{output_folder}/regression_line.png")
plt.show()
