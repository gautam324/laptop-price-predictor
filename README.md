# Laptop price predictor


# Laptop Price Prediction

This project aims to predict laptop prices using various machine learning models and provides a user interface for predictions. Below are the details on how the project is structured, the models used, and instructions for setting up and running the application.

## Models Used

1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **K-Nearest Neighbors (KNN)**
5. **Decision Tree**
6. **Support Vector Machine (SVM)**
7. **Random Forest**
8. **Extra Trees**
9. **AdaBoost**
10. **Gradient Boosting**
11. **XGBoost**
12. **Voting Regressor**
13. **Stacking Regressor**

### Performance Metrics
Each model's performance was evaluated using the following metrics:
- **R2 Score**
- **Mean Absolute Error (MAE)**

## Project Structure

The project consists of the following key files:
- `app.py`: The Streamlit application for predicting laptop prices.
- `pipe.pkl`: The trained machine learning model.
- `df.pkl`: The dataframe used for model training.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/laptop-price-prediction.git
   cd laptop-price-prediction
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Launch the Streamlit application:**
   Open your browser and navigate to `http://localhost:8501` to access the laptop price prediction interface.

2. **Fill in the details:**
   - Brand
   - Type of laptop
   - RAM
   - Weight
   - Touchscreen
   - IPS
   - Screen Size
   - Screen Resolution
   - CPU
   - HDD
   - SSD
   - GPU
   - OS

3. **Predict the Price:**
   Click on the 'Predict Price' button to get the predicted price for the specified laptop configuration.

## Model Training

### Data Preprocessing

A `ColumnTransformer` was used to preprocess the data, which included one-hot encoding for categorical features and passing through numerical features.

### Model Pipelines

For each model, a pipeline was created with two steps:
1. **Step 1:** Data transformation using `ColumnTransformer`.
2. **Step 2:** Model fitting using one of the regression algorithms.


## Results

| Model                 | R2 Score | MAE               |
|-----------------------|----------|-------------------|
| Linear Regression     | 0.807    | 0.210             |
| Ridge Regression      | 0.813    | 0.209             |
| Lasso Regression      | 0.807    | 0.211             |
| KNN                   | 0.802    | 0.193             |
| Decision Tree         | 0.847    | 0.181             |
| SVM                   | 0.808    | 0.202             |
| Random Forest         | 0.887    | 0.159             |
| Extra Trees           | 0.875    | 0.160             |
| AdaBoost              | 0.793    | 0.233             |
| Gradient Boosting     | 0.882    | 0.159             |
| XGBoost               | 0.881    | 0.165             |
| Voting Regressor      | 0.890    | 0.158             |
| Stacking Regressor    | N/A      | N/A               |


