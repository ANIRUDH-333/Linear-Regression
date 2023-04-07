A detailed code for single and multi variable linear regression with Gradient descent and in-built are implemented in repository.
Mainly answered all the [questions](https://github.com/ANIRUDH-333/Linear-Regression/blob/main/questions.pdf) in this [notebook](https://github.com/ANIRUDH-333/Linear-Regression/blob/main/Linear%20Regression.ipynb)

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
$\hat{y_i}$ is the predicted value of the target variable for the ith training example \
$y_i$ is the actual value of the target variable for the ith training example

We try to minimize the cost function typically using an optimization algorithm such as gradient descent algorithm.

### Gradient descent algorithm

Gradient descent is an optimization algorithm used to minimize the cost function of a machine learning model. It is a first-order iterative optimization algorithm that works by iteratively adjusting the model parameters in the direction of the negative gradient of the cost function, until the minimum of the cost function is reached.

The basic idea of gradient descent is to start with an initial set of parameter values, and then repeatedly update the parameters in the direction of the steepest descent of the cost function. The direction of steepest descent is given by the negative gradient of the cost function with respect to the parameters. The gradient is a vector that points in the direction of the greatest increase of the cost function, so taking the negative of the gradient points in the direction of the greatest decrease of the cost function.


The algorithm works by iteratively updating the parameters according to the following rule:

$$ b = b - \alpha∇L(b) $$

$$ \Rightarrow b_j = b_j - \alpha \frac{\partial } {\partial b_j} L(b) $$

$$ \Rightarrow b_j = b_j - \frac{\alpha}{m} \Sigma_{i=1}^{m} [(\hat{y_i} - y_i)x_i] $$

where:

b is the vector of model parameters, the coefficients and intercept in the case of linear regression \
α is the learning rate, which controls the size of the step taken in the direction of the negative gradient \
∇L(b) is the gradient of the cost function with respect to the parameters b \
<br>
The algorithm continues to iterate until the parameters converge to a minimum of the cost function or a maximum number of iterations is reached.

Contribute
I would be happy to resolve any queries related to this and open to any additions or changes in the repo.
