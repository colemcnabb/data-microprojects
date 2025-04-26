import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('telco_churn.csv')

print("Shape of Data: ",df.shape)

print("\nDataFrame Info: ")
df.info()

print("\nFirst 5 rows: ")
print(df.head(5))

# check for null values
print("\nMissing values by column: ")
df.isnull().sum().sort_values(ascending=False)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Create tenure groups
bins = [0,6,12,24,48,72,78]
labels = ['0-6', '7-12', '13-24', '25-48', '49-72', '73-78']
df['TenureGroup'] = pd.cut(df['tenure'], bins = bins, labels = labels, right = False)

print("\nTenure group value counts: ")
print(df['TenureGroup'].value_counts())

print("Missing by column:\n", df.isnull().sum())

df.info()

text_columns = df.select_dtypes(include= 'object')

for col in text_columns.columns:
    blanks = (df[col].str.strip() == '').sum()
    if blanks > 0:
        print(f"{col}: {blanks} blank or whitespace-only values")
        
for col in text_columns.columns:
    df[col] = df[col].str.strip()

# Check for duplicates
print("\nDuplicate values: ", df.duplicated().sum())

df.describe()

print("\nUnique values per categorical column:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\nTarget Distribution: ", df['Churn'].value_counts(normalize=True))

# Drop unnecessary column
df.drop('customerID', axis = 1, inplace = True)

df.info()

df.to_csv('clean_telco.csv', index=False)

# Create Average Monthly Charges by Tenure Group
sns.barplot(x = 'TenureGroup', y = 'MonthlyCharges', data = df, ci = None)
plt.title('Average Monthly Charges by Tenure Group')
plt.ylabel('Avg Monthly Charges')
plt.xlabel('Tenure Group (in months)')
plt.savefig('charges_by_group.png')
plt.show()

# Create Churn Rate by Contract Type
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
grouped = df.groupby('Contract')['Churn'].mean().reset_index()
grouped['Churn'] *= 100
sns.barplot(x = 'Contract', y = 'Churn', data = grouped,ci = None)
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Contract Type')
plt.savefig('churn_by_contract.png')
plt.show()

