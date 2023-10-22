import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif, RFE, RFECV, SelectFromModel, SelectPercentile, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Load ADNI dataset
adni_df = pd.read_csv('AD_train.csv')

# Split features and target variable
X = adni_df.drop('Diagnosis', axis=1)
y = adni_df['Diagnosis']

########### Encoding #################################
# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

########### END Encoding #################################
# define feature ranking techniques
k_best = SelectKBest(score_func=f_classif, k=5)
mutual_info = SelectKBest(score_func=mutual_info_classif, k=5)
chi_square = SelectKBest(score_func=chi2, k=5)
model_based = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=5)
recursive_elimination = RFECV(LogisticRegression(), cv=5)
forward_selection = SequentialFeatureSelector(LinearSVC(), n_features_to_select=5, direction='forward')
backward_selection = SequentialFeatureSelector(LinearSVC(), n_features_to_select=5, direction='backward')
bidirectional_selection = SequentialFeatureSelector(LinearSVC(), n_features_to_select=5, direction='bidirectional')
lasso = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000), max_features=5)
ridge = SelectFromModel(LogisticRegression(penalty='l2', solver='liblinear', max_iter=10000), max_features=5)

# apply feature ranking techniques
k_best.fit(X, y)
mutual_info.fit(X, y)
chi_square.fit(X, y)
model_based.fit(X, y)
recursive_elimination.fit(X, y)
lasso.fit(X, y)
ridge.fit(X, y)

# get selected features from each technique
k_best_features = X.columns[k_best.get_support()]
mutual_info_features = X.columns[mutual_info.get_support()]
chi_square_features = X.columns[chi_square.get_support()]
model_based_features = X.columns[model_based.get_support()]
recursive_elimination_features = X.columns[recursive_elimination.support_]
lasso_features = X.columns[lasso.get_support()]
ridge_features = X.columns[ridge.get_support()]

# print selected features for each technique
print('SelectKBest: ', k_best_features)
print('Mutual Information: ', mutual_info_features)
print('Chi-Square: ', chi_square_features)
print('Model-based: ', model_based_features)
print('Recursive Elimination: ', recursive_elimination_features)
print('lasso_features: ', lasso_features)
print('ridge_features: ', ridge_features)
