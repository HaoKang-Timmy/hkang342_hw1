import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel='linear', max_iters=5, tol=1e-3):
        self.C = C  # Regularization parameter
        self.kernel = kernel  # Kernel function: 'linear' or 'rbf'
        self.max_iters = max_iters  # Maximum number of iterations
        self.tol = tol  # Tolerance for stopping criterion
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        
        # Initialize alpha (Lagrange multipliers) and bias
        self.alpha = np.zeros(self.n_samples)
        self.bias = 0
        
        # Select the kernel function
        if self.kernel == 'linear':
            self.kernel_function = self.linear_kernel
        elif self.kernel == 'rbf':
            self.kernel_function = self.rbf_kernel
        else:
            raise ValueError("Unsupported kernel function")
        
        # Train the SVM
        self._train()
        
    def _train(self):
        for _ in range(self.max_iters):
            num_changed_alphas = 0
            print(_)
            for i in range(self.n_samples):
                if(i%1000==0):
                    print(i)
                Ei = self._predict(self.X[i]) - self.y[i]
                if (self.y[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
                   (self.y[i] * Ei > self.tol and self.alpha[i] > 0):
                    j = self._select_random_index(i)
                    Ej = self._predict(self.X[j]) - self.y[j]
                    
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    L, H = self._compute_L_H(self.alpha[i], self.alpha[j], self.y[i], self.y[j])
                    
                    if L == H:
                        continue

                    eta = 2.0 * self._kernel(self.X[i], self.X[j]) - self._kernel(self.X[i], self.X[i]) - self._kernel(self.X[j], self.X[j])
                    if eta >= 0:
                        continue

                    self.alpha[j] = self.alpha[j] - (self.y[j] * (Ei - Ej)) / eta
                    self.alpha[j] = self._clip_alpha(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.bias - Ei - self.y[i] * (self.alpha[i] - alpha_i_old) * self._kernel(self.X[i], self.X[i]) \
                         - self.y[j] * (self.alpha[j] - alpha_j_old) * self._kernel(self.X[i], self.X[j])
                    b2 = self.bias - Ej - self.y[i] * (self.alpha[i] - alpha_i_old) * self._kernel(self.X[i], self.X[j]) \
                         - self.y[j] * (self.alpha[j] - alpha_j_old) * self._kernel(self.X[j], self.X[j])

                    if 0 < self.alpha[i] < self.C:
                        self.bias = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.bias = b2
                    else:
                        self.bias = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                break

    def _predict(self, x):
        predictions = 0
        for i in range(self.n_samples):
            predictions += self.alpha[i] * self.y[i] * self._kernel(x, self.X[i])
        return predictions + self.bias

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = self._predict(x)
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
    
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1, x2, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    
    def _kernel(self, x1, x2):
        return self.kernel_function(x1, x2)
    
    def _select_random_index(self, i):
        j = i
        while j == i:
            j = np.random.randint(0, self.n_samples)
        return j
    
    def _compute_L_H(self, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        return L, H
    
    def _clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha
