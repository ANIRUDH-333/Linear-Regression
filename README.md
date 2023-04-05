# Linear-Regression
Have implemented the linear regression algorithm using the [in-built](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) (OLS method) and also using the Gradient Descent algorithm

### Model Representation

$$ y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n + ε $$

where y is the predicted output variable. $(b_1 - b_n)$ are the weights given to the input features $(x_1 - x_n)$ respectively. $b_0$ is the intercept. 
So this lies in $(n+1)^{th}$ dimensional space. 

The goal of linear regression is to find the values of the coefficients $(b_0 - b_n)$ that minimize the difference between the predicted values $(y)$ and the actual values $(x_1 - x_n)$ of the target variable. This is typically done using a method called least squares, which minimizes the sum of the squared errors between the predicted values and the actual values.

### Cost Function

$$ L(b_0, b_1, ..., b_n) = \frac{1}{2m} \Sigma_{i=1}^{m} (\hat{y_i} - y_i)^2 $$

where m is the number of training examples \
$ \hat{y_i} $ is the predicted value of the target variable for the ith training example \
$ y_i $ is the actual value of the target variable for the ith training example

We try to minimize the cost function typically using an optimization algorithm such as gradient descent algorithm.
