import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class NaiveBayes:
 def fit(self, X, y):
     n_samples, n_features = X.shape
     self.classes = np.unique(y)
     n_classes = len(self.classes)

     
     self.mean   = np.zeros((n_classes, n_features), dtype=np.float64)
     self.var    = np.zeros((n_classes, n_features), dtype=np.float64)
     self.priors = np.zeros(n_classes, dtype=np.float64)

     for idx, c in enumerate(self.classes):
         X_c = X[y == c]
         self.mean[idx, :]   = X_c.mean(axis=0)
         self.var[idx, :]    = X_c.var(axis=0) + 1e-9  
         self.priors[idx]    = X_c.shape[0] / n_samples

 def _pdf(self, class_idx, x):
     """Gaussian probability density function."""
     mean = self.mean[class_idx]
     var  = self.var[class_idx]
  
     num = np.exp(- (x - mean) ** 2 / (2 * var))
    
     den = np.sqrt(2 * np.pi * var)
     return num / den

 def _predict(self, x):
    
     posteriors = []
     for idx, c in enumerate(self.classes):
         prior_log = np.log(self.priors[idx])
         cond_log  = np.sum(np.log(self._pdf(idx, x)))
         posteriors.append(prior_log + cond_log)
    
     return self.classes[np.argmax(posteriors)]

 def predict(self, X):
    
     return np.array([self._predict(x_i) for x_i in X])


if __name__ == "__main__":
 def accuracy(y_true, y_pred):
     return np.mean(y_true == y_pred)


 X, y = datasets.make_classification(
     n_samples=1000, n_features=20,
     n_classes=2, n_informative=2,
     n_redundant=10, random_state=42
 )
 X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
 )

 model = NaiveBayes()
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 print(f"Accuracy: {accuracy(y_test, y_pred):.3f}")
