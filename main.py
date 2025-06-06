"""
We provide this file as an example of how to use the LogisticRegressor class
with the NSCLC dataset. You can use this both to test your implementation
and as a guide for how to use the NSCLC dataset in your unit test(s). You
don't necessarily need to change anything in here (aside from uncommenting)
unless you want to.
"""

# Imports
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define main function
def main():

    #Load data
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
  
    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    #print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    # For testing purposes, once you've added your code.
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=10, batch_size=10)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()

# Run main function if run as script
if __name__ == "__main__":
    main()
