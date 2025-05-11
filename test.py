# %% [markdown]
# # modeling

# %%
import numpy as np
import pandas as pd
import joblib as jl
import xgboost as xgb
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit ,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor

# %%
data = pd.read_csv("cleaned_dataset.csv")

# %%
data.info()

# %%
data.drop(columns=['Unnamed: 0'], inplace=True)

# %% [markdown]
# # Define features and target
# 

# %%
# Define features and target
X = data.drop(columns=['Unit Price'])
y = data['Unit Price']

# %% [markdown]
# This line saves the `X.columns` list to a file using `joblib` (imported as `jl`):
# 
# - `X.columns`: The list of feature names used for model training or inference.
# - `'deploy/columns.pkl'`: The file path where the list will be saved in **pickle** format.
# 
# This is useful for ensuring that the same column order and names are used during model deployment or future predictions.
# 

# %%
columns_list = X.columns.tolist()
jl.dump(columns_list, "deploy/columns.pkl", compress=3)

# %% [markdown]
# # The code snippet demonstrates the usage of `TimeSeriesSplit` from `scikit-learn` to perform **time-series cross-validation**.
# 
# - `TimeSeriesSplit(n_splits=6)`: Creates a time series splitter with 6 splits.
# - `tscv.split(X)`: Generates indices for training and testing sets, maintaining the order of time.
# - In each iteration:
#   - `X_train`, `X_test`: Training and testing features split based on the indices.
#   - `y_train`, `y_test`: Corresponding training and testing target variables.
# 
# Unlike standard cross-validation, **TimeSeriesSplit** ensures that the training set is always before the testing set, respecting the temporal order of data.
# 

# %%
tscv = TimeSeriesSplit(n_splits=6)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# %% [markdown]
# # The code snippet initializes an **XGBoost Regressor** model:
# 
# - `xgb.XGBRegressor`: This is the XGBoost implementation for regression tasks.
# - `objective='reg:squarederror'`: Specifies the regression loss function as **squared error**.
# - `n_jobs=2`: Utilizes 2 CPU cores for parallel processing.
# - `random_state=42`: Sets a random seed for reproducibility of results.
# 
# XGBoost is an efficient and powerful gradient boosting algorithm, widely used for its performance and speed.
# 

# %%
model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=2, random_state=0)

# %% [markdown]
# # The code snippet defines a dictionary of hyperparameters for tuning the **XGBoost Regressor**:
# 
# - `n_estimators`: The number of boosting rounds (iterations), with values `[100, 300, 500, 800, 1000]`.
# - `learning_rate`: The step size shrinkage used to prevent overfitting, with values `[0.01, 0.05, 0.1, 0.3]`.
# - `max_depth`: The maximum depth of each tree, with values `[3, 5, 7, 9]`.
# - `subsample`: The fraction of samples used for fitting individual base learners, with values `[0.6, 0.8, 1.0]`.
# - `reg_alpha`: L1 regularization term (Lasso) to reduce overfitting, with values `[0, 0.1, 0.5]`.
# - `reg_lambda`: L2 regularization term (Ridge) to control model complexity, with values `[0.5, 1, 2]`.
# 
# This parameter grid is typically used for hyperparameter optimization techniques like **GridSearchCV** or **RandomizedSearchCV**.
# 

# %%
param_dist = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 2]
}

# %% [markdown]
# # The code snippet initializes a **RandomizedSearchCV** for hyperparameter optimization:
# 
# - `model`: The XGBoost regressor instance to be optimized.
# - `param_distributions=param_dist`: The hyperparameter grid to sample from, defined earlier.
# - `n_iter=50`: The number of parameter settings that are sampled, reducing search time compared to exhaustive search.
# - `scoring='neg_root_mean_squared_error'`: The evaluation metric used to measure model performance (lower is better).
# - `cv=tscv`: The cross-validation strategy, using **TimeSeriesSplit** to respect the temporal order of data.
# - `verbose=1`: Displays progress logs during the search process.
# 
# **RandomizedSearchCV** efficiently explores the hyperparameter space by sampling a subset of combinations, speeding up the search compared to **GridSearchCV**.
# 

# %%
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50,  
    scoring='neg_root_mean_squared_error',
    cv=tscv
)

# %% [markdown]
# # The code snippet executes the hyperparameter optimization:
# - Fits the **RandomizedSearchCV** instance to the training data (`X`, `y`).
# - Iterates through 50 different hyperparameter combinations (as defined earlier) using **time-series cross-validation**.
# - Evaluates each combination based on the **negative root mean squared error** (neg\_rmse).
# 
# After completion, the model is optimized with the best-found hyperparameters, accessible through:
# -  Displays the best parameter combination.
# -  Shows the best score achieved during the search.
# -  Provides the fully trained model with the optimal parameters.
# 

# %%
random_search.fit(X, y)

# %% [markdown]
# # The code snippet selects and trains the best model found during hyperparameter optimization:
# 
# -  Retrieve the XGBoost model instance with the optimal hyperparameters discovered during **RandomizedSearchCV**.
# -  Trains the optimized model on the training dataset (`X_train`, `y_train`).
# 
# At this stage, the model is fully optimized and ready for evaluation or prediction.
# 

# %%
model = random_search.best_estimator_
model.fit(X_train, y_train)

# %% [markdown]
# # Predict on training and test data
# 
# 

# %%
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# %% [markdown]
# # Evaluate the model on training & test data
# 
# 

# %%
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

# %%
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

# %% [markdown]
# # Print evaluation metrics
# 

# %%
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Train R² Score: {r2_train:.4f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test R² Score: {r2_test:.4f}")

# %% [markdown]
# # Random Forest Regressor

# %% [markdown]
# ### Training a Random Forest Regressor
# - `RandomForestRegressor`: An ensemble learning method that builds multiple decision trees and averages their predictions for regression tasks.
# - `n_estimators=100`: Builds 100 trees in the forest.
# - `random_state=0`: Ensures reproducibility by controlling randomness.
# - `n_jobs=2`: Runs training using 2 CPU cores in parallel.
# - `max_depth=7`: Limits the depth of each tree to prevent overfitting.
# - `max_samples=0.8`: Each tree is trained on a random 80% subset of the data (row sampling).

# %%
rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=2,max_depth=7,max_samples=0.8)
rf_model.fit(X_train, y_train)

# %% [markdown]
# ## Making Predictions with the Trained Model
# - `rf_model.predict(X_train)`: Generates predictions on the training set to evaluate how well the model fits the training data.
# - `rf_model.predict(X_test)`: Generates predictions on the test set to assess the model's generalization performance.
# 
# These predicted values (y_train_pred, y_test_pred) can be used to compute evaluation metrics such as RMSE, MAE, or R².

# %%
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# %% [markdown]
# ### Evaluating Model Performance
# RMSE (Root Mean Squared Error): Measures the average magnitude of the errors in the model’s predictions. It is the square root of the mean squared error.
# - `rmse_train`: RMSE for predictions on the training data.
# - `rmse_test`: RMSE for predictions on the test data.
# - `R² (R-squared)`: Indicates the proportion of the variance in the target variable that is explained by the model. A higher R² value indicates a better fit.
# - `r2_train`: R² score for predictions on the training data.
# - `r2_test`: R² score for predictions on the test data.
# 
# These metrics are used to assess the model's accuracy and its ability to generalize to unseen data.

# %%
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

# %% [markdown]
# # Displaying Random Forest Model Results

# %%
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Train R² Score: {r2_train:.4f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test R² Score: {r2_test:.4f}")

# %% [markdown]
# ### Initializing a Gradient Boosting Regressor Model
# GradientBoostingRegressor: A boosting ensemble algorithm that builds trees sequentially, where each tree corrects errors of the previous one.
# - `n_estimators=100`: The number of boosting stages (trees) to be built. Here, 100 trees are used.
# - `learning_rate=0.1`: The step size at each iteration, controlling how much the model learns from each tree.
# - `max_depth=3`: Limits the depth of each individual tree to prevent overfitting.
# - `random_state=0`: Ensures reproducibility by controlling randomness during training.
# 
# This model is suitable for regression tasks where boosting can improve accuracy over individual decision trees.

# %%
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)

# %% [markdown]
# # Fit the model to the training data
# 

# %%
# Fit the model to the training data
gbr_model.fit(X_train, y_train)

# %% [markdown]
# # Making Predictions with Gradient Boosting Regressor
# - Predicts target values on the training set.
# - Predicts target values on the test set.
# 
# These predictions will later be used to evaluate the model's accuracy and generalization.

# %%
y_train_pred = gbr_model.predict(X_train)
y_test_pred = gbr_model.predict(X_test)

# %% [markdown]
# # Evaluating Gradient Boosting Regressor Performance
# - Measures the average magnitude of prediction errors. Lower is better.
# - Indicates how well predictions approximate actual values. Ranges from 0 to 1, with higher values indicating better fit.
# 
# These metrics help determine how well the Gradient Boosting model performs on both training and unseen test data.

# %%
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

# %% [markdown]
# # Displaying Gradient Boosting Model Results

# %%
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Train R² Score: {r2_train:.4f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test R² Score: {r2_test:.4f}")

# %% [markdown]
# ### Saving Trained Models
# 

# %%
jl.dump(model, 'deploy/xgb_model.pkl')
jl.dump(rf_model, 'deploy/rf_model.pkl')
jl.dump(gbr_model, 'deploy/gbr_model.pkl')

# %%
# Streamlit input widgets
tv = st.slider("TV Ad Budget", 0, 300, 150)
radio = st.slider("Radio Ad Budget", 0, 50, 25)
newspaper = st.slider("Newspaper Ad Budget", 0, 100, 50)

# Prediction
if st.button("Predict Sales"):
    prediction = model.predict([[tv, radio, newspaper]])
    st.success(f"Predicted Sales: ${prediction[0]:.2f}")


