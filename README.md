# Housing Price Prediction (Linear Regression from scratch)

A Python project comparing a from-scratch multivariate linear regression model against scikit-learn using a housing prices dataset.

## Features

* **Implementation from scratch:** Built gradient descent, MSE loss tracking, feature scaling (Z-score normalisation), and $R^2$ score calculation using NumPy.
* **Scikit-Learn baseline:** Used scikit-learn's `LinearRegression` model for comparison.
* **Data Processing:** Cleaned categorical columns using Pandas.
* **Visualisation:** Plotted actual vs. predicted prices using Matplotlib.
* **Evaluation:** Made function to calculate MSE, MAE, and $R^2$ on different models with different test sizes.

## Results (Linear Regression)

* **Manual (NumPy):** $R^2 \approx 0.674$
* **Scikit-Learn:** $R^2 \approx 0.673$
