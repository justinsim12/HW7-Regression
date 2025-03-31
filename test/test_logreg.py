"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

import pytest
import numpy as np

# Assuming your logistic regression implementation is in logreg.py
# and that it has a class 'LogisticRegression' with methods:
#   - predict(x): returns the prediction for a given input x
#   - loss(X, y): computes the binary cross-entropy loss for dataset (X, y)
#   - gradient(X, y): returns a tuple (grad_w, grad_b) of gradients
#   - train(X, y, iterations, learning_rate): trains the model on (X, y)
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler


def test_prediction():
    # Create a LogisticRegressor object
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
    # Reset the model
    log_model.reset_model()
    # Create a random input and append a bias term
    X = np.append(np.random.rand(6), 1)
    # Predict the output
    y_pred = log_model.make_prediction(X)
    # Check that the output is between 0 and 1
    assert 0 <= y_pred <= 1

def test_loss_function():
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
    log_model.reset_model()
    
    # Set weights to known values for a deterministic test
    log_model.W = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
    
    # Create two examples: one expected to be class 1 and one class 0
    X1 = np.append(np.ones(6), 1)   # all ones (with bias term)
    X2 = np.append(np.zeros(6), 1)  # all zeros (with bias term)
    X = np.vstack([X1, X2])
    y = np.array([1, 0])
    
    # Manually compute predictions using the logistic function
    p1 = 1 / (1 + np.exp(-np.dot(X1, log_model.W)))
    p2 = 1 / (1 + np.exp(-np.dot(X2, log_model.W)))
    
    # Compute binary cross-entropy loss for each sample
    loss1 = - (1 * np.log(p1) + (1 - 1) * np.log(1 - p1))
    loss2 = - (0 * np.log(p2) + (1 - 0) * np.log(1 - p2))
    expected_loss = (loss1 + loss2) / 2
    
    # Compute loss using the loss_function method
    computed_loss = log_model.loss_function(y, log_model.make_prediction(X))
    
    assert np.isclose(computed_loss, expected_loss, atol=1e-4)

def test_gradient():
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
    log_model.reset_model()
    
    # Set weights to known values
    log_model.W = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
    
    # Create a single sample input (with bias term)
    X = np.append(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 1)
    # Reshape to 2D as required by calculate_gradient
    X_batch = X.reshape(1, -1)
    y_val = np.array([1])
    
    # Manually compute prediction
    z = np.dot(X, log_model.W)
    p = 1 / (1 + np.exp(-z))
    # For a single sample, error is p - y (here y=1)
    error = p - 1
    
    # Expected gradient: (1/N)*X^T * (y_pred - y_true); here N = 1 so it is error * X
    expected_grad = error * X
    
    computed_grad = log_model.calculate_gradient(y_val, X_batch)
    
    assert np.allclose(computed_grad, expected_grad, atol=1e-4)

def test_training():
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.001, max_iter=100, batch_size=2)
    log_model.reset_model()
    
    # Create a simple dataset with two examples for each class (without bias term)
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
  
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10000, batch_size=10)
    log_model.train_model(X_train, y_train, X_val, y_val)   
    
    # Check that the loss is decreasing
    assert log_model.loss_hist_train[-1] < log_model.loss_hist_train[0]
    assert log_model.loss_hist_val[-1] < log_model.loss_hist_val[0]
    