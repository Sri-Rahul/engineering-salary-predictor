# Engineering Salary Predictor

## Overview
This project implements a machine learning-based salary prediction system for engineering graduates in India. The model uses features such as college state, specialization, and graduation year to predict potential salaries using multiple regression algorithms.

## Features
- Multiple prediction models:
  - Linear Regression
  - Random Forest Regression
- Automated data preprocessing pipeline
- Support for both numerical and categorical features
- Interactive command-line interface
- Comprehensive error handling
- Model performance metrics (RMSE)

## Prerequisites
```
Python 3.7+
pandas
numpy
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sri-Rahul/engineering-salary-predictor.git
cd engineering-salary-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Requirements
The program expects a CSV file named 'GKD_JOBS_ALL_ENGG.csv' with the following columns:
- CollegeState: State where the college is located
- Specialization: Engineering specialization/branch
- GraduationYear: Year of graduation
- Salary: Annual salary in INR

## Usage

1. Prepare your dataset:
   - Ensure your CSV file follows the required format
   - Place it in the project directory

2. Run the program:
```bash
python salary_predictor.py
```

3. Follow the interactive prompts:
   - Select your specialization from the displayed list
   - Enter your graduation year
   - View predicted salaries from different models

## Code Structure

### SalaryPredictor Class
```python
class SalaryPredictor:
    def __init__(self)
    def preprocess_data(self)
    def fit(self, X, y)
    def predict(self, X)
```

- `__init__`: Initializes model configurations and preprocessing pipelines
- `preprocess_data`: Sets up data transformation pipeline
- `fit`: Trains models on the provided data
- `predict`: Generates salary predictions using trained models

## Data Preprocessing
The system automatically handles:
- Standardization of numerical features
- One-hot encoding of categorical variables
- Missing value handling
- Feature scaling

## Model Details

### Linear Regression
- Basic linear regression model
- Useful for understanding linear relationships in the data
- Provides baseline predictions

### Random Forest Regressor
- Ensemble learning method
- Handles non-linear relationships
- More robust to outliers
- Random state fixed for reproducibility

## Error Handling
The system includes comprehensive error handling for:
- Invalid input values
- Out-of-range specialization selections
- Data format issues
- General exceptions

## Performance Metrics
The system outputs Root Mean Square Error (RMSE) for each model during training, allowing for model performance comparison.

## Sample Output
```
=== Salary Predictions ===
Specialization: Computer Science
Graduation Year: 2022
Linear Regression Predicted Salary: ₹800,000.00
Random Forest Predicted Salary: ₹850,000.00
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Dataset source: [[dataset source here](https://github.com/muskaanpirani/Jobs_and_admission_prediction_SIH2020)]
- Inspired by the need for transparent salary predictions in the engineering sector
- Built using scikit-learn's comprehensive machine learning tools

## Contact
Project Link: [https://github.com/Sri-Rahul/engineering-salary-predictor]
