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

class SalaryPredictor:
    def __init__(self):
        self.numeric_features = ['GraduationYear']
        self.categorical_features = ['CollegeState', 'Specialization']
        self.preprocessor = None
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        self.fitted_models = {}
        
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
        self.preprocess_data()
        for name, model in self.models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X, y)
            self.fitted_models[name] = pipeline
            y_pred = pipeline.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            print(f"{name} RMSE: {rmse:.2f}")
            
    def predict(self, X):
        predictions = {}
        for name, model in self.fitted_models.items():
            pred = model.predict(X)
            predictions[name] = pred[0]
        return predictions

def main():
    df = pd.read_csv('/content/GKD_JOBS_ALL_ENGG.csv')
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