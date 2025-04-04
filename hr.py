import pandas as pd

df = pd.read_csv("hr.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # To check for missing values
df = df.drop(columns=['EmployeeCount', 'Over18', 'EmployeeNumber'], errors='ignore')
# Drop rows with any nulls (if few):
df = df.dropna()

# Or fill missing values (if many):
# df['ColumnName'] = df['ColumnName'].fillna(df['ColumnName'].median())
df = pd.get_dummies(df, drop_first=True)  # One-hot encoding for categorical variables
# For example, remove rows where MonthlyIncome is in the top 1%
df = df[df['MonthlyIncome'] < df['MonthlyIncome'].quantile(0.99)]
df.to_csv("cleaned_employee_data.csv", index=False)
