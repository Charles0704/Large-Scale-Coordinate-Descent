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


$\mathcal{L}(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( y_i - \sum_{j=0}^{n} \theta_j x_{ij} \right)^2 + \lambda \sum_{j=0}^{n} |\theta_j|\$


where $\lambda$ is the regularization strength parameter, which controls the sparsity of the model.


## 3. METHODOLOGY  
Coordinate descent is an optimization algorithm used to minimize multivariate functions by iteratively solving simpler univariate problems. At each iteration, the algorithm selects a single coordinate direction and minimizes the objective function along that direction, while keeping all other coordinates fixed. This process repeats cyclically for each coordinate until convergence. Given an initial point \( \mathbf{x}^{(0)} = (x_1^0, x_2^0, \dots, x_n^0) \), the algorithm updates each coordinate by solving:

$x_i^{(k+1)} = \arg \min_{x_i} f(x_1^{(k+1)}, \dots, x_{i-1}^{(k+1)}, x_i, x_{i+1}^{(k)}, \dots, x_n^{(k)})\$

Alternatively, a gradient-based update can be used for \( x_i \), such that:

$x_i := x_i - \alpha \frac{\partial f}{\partial x_i}(\mathbf{x})\$

where $\( \alpha \)$ is a step size parameter. This method is particularly efficient for high-dimensional problems, as each iteration focuses on a simpler, one-dimensional subproblem.

### 3.2. Coordinate Descent For LASSO
The first equation defines the $\textbf{soft thresholding function}, \S(\alpha, \lambda) \$, which is widely used in Lasso regression for variable selection:  

$\S(\alpha, \lambda) =\begin{cases}\alpha - \lambda & \text{if } \alpha > \lambda \\0 & \text{if } |\alpha| \leq \lambda \\\alpha + \lambda & \text{if } \alpha < -\lambda\end{cases}\$

For easier computation, we transform the soft shresholding function into the following form:


$\S(\alpha, \lambda) = \text{sign}(\alpha) \max(|\alpha| - \lambda, 0)\$

The soft thresholding operator shrinks the coefficient $\alpha\$ by $\lambda\$, setting it to zero if it falls below the threshold.

The second equation, 

$\rho_j = \sum_{i=1}^{m} x_{j}^i \left( y_i - \sum_{k \neq j}^{n} \theta_k x_{k}^i \right)\$

is the $\textbf{partial residual sum}$ used in coordinate descent for Lasso regression. Here, $\rho_j\$ is computed for each feature $\\ x_j \$, accounting for the current values of the other coefficients $\\theta_k \ for \( k \neq j \)$.

Finally, the third equation rewrites the residual sum by isolating the contribution of $\( x_j \)$ to the prediction:

$\rho_j = \sum_{i=1}^{m} x_{j}^i \left( y_i -\hat{y}_pred^i+\theta_k x_{j}^i \right)\$

where $\(\hat{y}_{\text{pred}}^i\)$ is the predicted value excluding the contribution from the current feature $\( x_j \)$. This reformulation highlights the effect of updating $\(\theta_j\)$ on the residual. Algorithm 1 shows the pseudocode for lasso coordinate descent.


## 4. EXPERIMENT  










