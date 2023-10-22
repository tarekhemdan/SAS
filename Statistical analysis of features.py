import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Load ADNI dataset as a Pandas dataframe
adni_df = pd.read_csv('AD_train.csv')

# Encode the sex variable
le = LabelEncoder()
adni_df['sex'] = le.fit_transform(adni_df['sex'])

# Print the number of samples and features in the dataset
print("Number of samples: ", adni_df.shape[0])
print("Number of features: ", adni_df.shape[1] - 1) # Exclude the sex variable

# Compute descriptive statistics for each feature
for col in adni_df.columns[:-1]: # Exclude the sex variable
    stat_dict = {
        'mean': adni_df[col].mean(),
        'median': adni_df[col].median(),
        'std_dev': adni_df[col].std(),
        'min': adni_df[col].min(),
        '25%': adni_df[col].quantile(0.25),
        '50%': adni_df[col].quantile(0.50),
        '75%': adni_df[col].quantile(0.75),
        'max': adni_df[col].max()
    }
    print(f"\n{col} statistics:")
    for key, value in stat_dict.items():
        print(f"{key}: {value:.2f}")

# Perform t-test between Normal and AD groups for sex feature
group1 = adni_df[adni_df['sex'] == 0]['sex']
group2 = adni_df[adni_df['sex'] == 1]['sex']
t_stat, p_val = stats.ttest_ind(group1, group2)
print(f"\nT-test results for sex feature: t-statistic = {t_stat:.2f}, p-value = {p_val:.2f}")