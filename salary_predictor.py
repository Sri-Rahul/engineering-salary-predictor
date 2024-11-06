import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class NonNegativeLinearRegression(LinearRegression):
    def predict(self, X):
        predictions = super().predict(X)
        return np.maximum(predictions, 0)  # Ensures predictions are non-negative

class SalaryPredictor:
    def __init__(self):
        self.numeric_features = ['GraduationYear']
        self.categorical_features = ['CollegeState', 'Specialization']
        self.preprocessor = None
        self.models = {
            'Linear Regression': NonNegativeLinearRegression(),
            'Random Forest': RandomForestRegressor(
                random_state=42,
                min_samples_leaf=2,  # prevent extreme predictions
                n_estimators=100
            )
        }
        self.fitted_models = {}
        self.y_scaler = StandardScaler()  # Add scaling

    def preprocess_data(self):
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def fit(self, X, y):
        # Apply log transformation to salary (prevent negative predictions)
        y_log = np.log1p(y)
        
        self.preprocess_data()
        for name, model in self.models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X, y_log)
            self.fitted_models[name] = pipeline
            y_pred_log = pipeline.predict(X)
            y_pred = np.expm1(y_pred_log)  # Transform predictions back to original scale
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            print(f"{name} RMSE: {rmse:.2f}")
            print(f"{name} R2 Score: {r2:.3f}")

    def predict(self, X):
        predictions = {}
        for name, model in self.fitted_models.items():
            pred_log = model.predict(X)
            pred = np.expm1(pred_log)  # Transform back to original scale
            predictions[name] = max(0, pred[0])  # Ensure prediction is non-negative
        return predictions

def main():
    df = pd.read_csv('/content/GKD_JOBS_ALL_ENGG.csv')
    
    # Remove outliers using IQR method
    Q1 = df['Salary'].quantile(0.25)
    Q3 = df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[
        (df['Salary'] >= Q1 - 1.5 * IQR) & 
        (df['Salary'] <= Q3 + 1.5 * IQR)
    ]
    
    predictor = SalaryPredictor()
    X = df[['CollegeState', 'Specialization', 'GraduationYear']]
    y = df['Salary']
    predictor.fit(X, y)
    
    specializations = df['Specialization'].unique()
    print("\nAvailable Specializations:")
    for i, spec in enumerate(specializations, 1):
        print(f"{i}. {spec}")
        
    try:
        spec_index = int(input("\nEnter the number corresponding to your specialization: ")) - 1
        graduation_year = int(input("Enter graduation year (e.g., 2015): "))
        
        if spec_index < 0 or spec_index >= len(specializations):
            raise ValueError("Invalid specialization number")
            
        specialization = specializations[spec_index]
        input_data = pd.DataFrame({
            'CollegeState': ['Delhi'],
            'Specialization': [specialization],
            'GraduationYear': [graduation_year]
        })
        
        predictions = predictor.predict(input_data)
        print("\n=== Salary Predictions ===")
        print(f"Specialization: {specialization}")
        print(f"Graduation Year: {graduation_year}")
        for name, salary in predictions.items():
            print(f"{name} Predicted Salary: ₹{salary:,.2f}")
            
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Please enter valid numeric inputs")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
