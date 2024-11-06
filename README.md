## 1.INTRODUCTION：  
Sparse regression has become highly valuable across a wide range of applications due to its enhanced model interpretability and ability
to improve efficiency by selecting only the most relevant features.
A core optimization technique in sparse regression is coordinate descent, particularly effective in methods like LASSO and Elastic Net,
which promote sparsity. This technique is essential for its computational efficiency, scalability, reliable convergence properties, and
suitability for handling non-differentiable functions. In this report,
we explore the application of coordinate descent in LASSO regression, a widely-used approach with convex loss function for achieving
sparse solutions. We also conduct experiments to implement coordinate descent for LASSO, and analyze the results.  

## 2.BACKGROUND  

### 2.1. Sparse Regression  
Sparse regression is a statistical approach that focuses on producing models with only a subset of the most relevant features, resulting in interpretable and efficient solutions. By encouraging sparsity—where many feature coefficients are zero or close to zero—sparse regression reduces complexity. The main idea of enhancing sparsity is based on $L_0$ regularization, which aims to control the number of non-zero elements in the model parameters to achieve sparsity. However, directly solving the $L_0$ regularization problem is NP-hard. Therefore, a binary mask $m \in \{0, 1\}$ is typically introduced to transform the problem into $L_1$ regularization\cite{jacobs2024mask} like LASSO. Specifically, the sparsity objective is turned into a discrete $L_1$ penalty of $m$:


$\theta \odot m \|_{L_0} = \sum_i m_i$

where $\theta$ represents the model parameters, $m$ is the mask, and $\odot$ denotes element-wise multiplication.

### 2.2. LASSO  
Lasso\cite{tibshirani1996regression} is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical model. Lasso adds a penalty term, called the L1 regularization term equal to the absolute value of the magnitude of coefficients to the loss function, which helps to avoid overfitting by shrinking the coefficients of less important features toward zero:


$\mathcal{L}(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( y_i - \sum_{j=0}^{n} \theta_j x_{ij} \right)^2 + \lambda \sum_{j=0}^{n} |\theta_j|$


where $\lambda$ is the regularization strength parameter, which controls the sparsity of the model.


## 3. METHODOLOGY  

## 4. EXPERIMENT  










