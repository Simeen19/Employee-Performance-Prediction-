import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# 1. Load cleaned data
df = pd.read_csv("/kaggle/input/cleaned-employee-data/cleaned_employee_data.csv")

# 2. Define classification and regression targets
target_class = 'Attrition_Yes' if 'Attrition_Yes' in df.columns else None
target_reg = 'PerformanceRating' if 'PerformanceRating' in df.columns else None

# 3. Split features and targets
X = df.drop(columns=[col for col in [target_class, target_reg] if col])
y_class = df[target_class] if target_class else None
y_reg = df[target_reg] if target_reg else None

# 4. Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42) if y_class is not None else (None, None, None, None)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42) if y_reg is not None else (None, None, None, None)

# 5. Train models
if y_class is not None:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    print("\n--- Classification Report (Attrition Prediction) ---")
    print(classification_report(y_test_c, y_pred_c))

    cm = confusion_matrix(y_test_c, y_pred_c)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

if y_reg is not None:
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    print("\n--- Regression Metrics (Performance Rating Prediction) ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.2f}")
    print(f"R² Score: {r2_score(y_test_r, y_pred_r):.2f}")

# 6. Feature importance
model = clf if y_class is not None else reg
importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10), palette='viridis')
plt.title("Top 10 Influencing Features")
plt.show()

# 7. Predict employee most likely to get promoted based on work hours
if y_reg is not None:
    feature_cols = X_train_r.columns  # Ensure we use the same features as training
    X_promotion = df[feature_cols]  # Select only the required columns

    df['Predicted_Promotion'] = reg.predict(X_promotion)  # Predict using the full dataset
    most_likely_promoted = df.sort_values(by='Predicted_Promotion', ascending=False).head(1)

    print("\n--- Employee Most Likely to be Promoted ---")
    print(most_likely_promoted[['Age', 'JobLevel', 'TotalWorkingYears', 'Predicted_Promotion']])

