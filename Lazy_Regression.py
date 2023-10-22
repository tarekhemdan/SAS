import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import TweedieRegressor, PoissonRegressor, LassoLars, HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor, OrthogonalMatchingPursuit, Lars

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Extract the 'sex' column
sex = df['sex']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'sex' column
sex_encoded = label_encoder.fit_transform(sex)

# Update the 'sex' column in the DataFrame
df['sex'] = sex_encoded

# Split the dataset into features and target variable
X = df.drop('Posttreatment SAS 90', axis=1)
y = df['Posttreatment SAS 90']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the LazyRegressor
regressor = LazyRegressor(verbose=0, ignore_warnings=True)

# Fit and predict with all available regression models
models, predictions = regressor.fit(X_train, X_test, y_train, y_test)

# Print regression metrics for all models
for model_name, metrics in models.items():
    print(f"Model: {model_name}")
    if 'r2_score' in metrics:
        print(f"R-squared: {metrics['r2_score']}")
    else:
        print("R-squared: N/A")
    if 'mse' in metrics:
        print(f"Mean Squared Error: {metrics['mse']}")
    else:
        print("Mean Squared Error: N/A")
    if 'mae' in metrics:
        print(f"Mean Absolute Error: {metrics['mae']}")
    else:
        print("Mean Absolute Error: N/A")
    print("-------------------------------")