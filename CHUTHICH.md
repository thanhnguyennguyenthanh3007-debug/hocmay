# House Price Prediction Using Machine Learning

## 1. Project Overview

This project applies machine learning techniques to predict house prices based on several features, including:

* House area in square meters
* Number of bedrooms
* Distance from the city center

The project uses synthetic house-price data and compares different regression models to identify a suitable model for house price prediction.

The main purpose is to demonstrate:

* Data generation and preprocessing
* Polynomial regression
* Overfitting
* K-Fold Cross-Validation
* Ridge Regression
* Model evaluation using MSE and MAE

## 2. Technologies and Libraries

The project is implemented in Python using the following libraries:

* **NumPy**: Used to generate numerical data and random values.
* **Pandas**: Used to create and process the house-price dataset.
* **Scikit-learn**: Used to build, train, validate, and evaluate machine learning models.

Main Scikit-learn components:

* `train_test_split`
* `KFold`
* `cross_val_score`
* `PolynomialFeatures`
* `StandardScaler`
* `LinearRegression`
* `Ridge`
* `make_pipeline`
* `mean_squared_error`
* `mean_absolute_error`

## 3. Dataset

The dataset is generated directly in the Python program.

The project creates data for **30 houses** with the following features:

| Feature          | Description                                 |
| ---------------- | ------------------------------------------- |
| `dien_tich_m2`   | House area in square meters                 |
| `so_phong`       | Number of bedrooms                          |
| `khoang_cach_km` | Distance from the city center in kilometers |
| `gia_nha_ty`     | House price in billions of Vietnamese đồng  |

The house price is generated using an assumed formula:

```text
House price = 0.05 × Area
           + 0.3 × Number of bedrooms
           - 0.15 × Distance
           + 1.2
           + Random noise
```

The target variable is expressed in **billions of Vietnamese đồng**.

## 4. Project Workflow

The program follows these main steps:

### Step 1: Generate the Dataset

The program generates house features such as area, number of bedrooms, and distance from the city center. It then calculates the house price and stores all information in a Pandas DataFrame.

### Step 2: Split the Dataset

The dataset is divided into two parts:

* **Training set**: 70% of the data
* **Testing set**: 30% of the data

The training set is used to train the models, while the testing set is used to evaluate their performance on unseen data.

### Step 3: Build an Overfitting Model

A third-degree polynomial regression model is created using:

* `StandardScaler`
* `PolynomialFeatures(degree=3)`
* `LinearRegression`

Because the dataset is small and the polynomial model creates many additional features, the model may overfit the training data.

### Step 4: Compare Candidate Models

The program compares four different models:

1. First-degree linear regression
2. Second-degree polynomial regression
3. Second-degree polynomial regression with Ridge regularization
4. Third-degree polynomial regression

The models are evaluated using **5-Fold Cross-Validation**.

### Step 5: Select the Best Model

The model with the lowest average validation Mean Squared Error is selected as the best model.

The selected model is then trained again using the training dataset.

### Step 6: Evaluate the Final Model

The selected model is evaluated on the test dataset using **Mean Absolute Error (MAE)**.

The program also displays a comparison between:

* Actual house prices
* Predicted house prices from the best model
* Prediction errors
* Predictions from the overfitting model

## 5. Machine Learning Models

### Linear Regression

Linear Regression models the relationship between the input features and house prices using a linear function.

### Polynomial Regression

Polynomial Regression extends linear regression by creating polynomial combinations of the input features. This allows the model to learn more complex relationships.

However, using a high polynomial degree with a small dataset can cause overfitting.

### Ridge Regression

Ridge Regression adds L2 regularization to reduce the size of model coefficients. It can help reduce overfitting and improve model generalization.

## 6. Model Evaluation

The project uses the following evaluation metrics:

### Mean Squared Error

Mean Squared Error, or MSE, measures the average squared difference between actual and predicted values.

A lower MSE generally indicates better prediction performance.

### Mean Absolute Error

Mean Absolute Error, or MAE, measures the average absolute difference between actual and predicted house prices.

In this project, MAE is expressed in billions of Vietnamese đồng.

## 7. How to Run the Project

### Step 1: Install Python

Make sure Python is installed on your computer.

### Step 2: Install Required Libraries

Open a terminal and run:

```bash
pip install numpy pandas scikit-learn
```

### Step 3: Run the Python File

Run the following command:

```bash
python gianha.py
```

## 8. Expected Output

The program displays:

* The first five rows of the generated house-price dataset
* The number of training and testing samples
* The training and testing MSE of the overfitting model
* The average validation MSE of each candidate model
* The best model selected by K-Fold Cross-Validation
* The final MAE on the testing dataset
* A table comparing actual and predicted house prices

## 9. Limitations

This project uses a synthetic dataset rather than real-world house-price data.

Therefore:

* The dataset is relatively small.
* The generated prices are based on an assumed formula.
* The results may not represent actual housing-market conditions.
* The model has not been tested on a large real-world dataset.

## 10. Future Improvements

Possible improvements include:

* Using a real house-price dataset
* Increasing the number of training samples
* Adding more features, such as house location, property age, and floor number
* Visualizing actual versus predicted prices
* Comparing additional machine learning algorithms
* Deploying the model as a web application using Flask or Streamlit

## 11. Author

This project was created for learning and practicing machine learning, regression models, and model evaluation using Python.
