
### Import libraries ###
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay

### Ignore warnings ###
warnings.filterwarnings('ignore')

### Random seed number for reproducibility ###
seed = 42
np.random.seed(seed)

#-----------------------------------------------#

### Loading the dataset ####
d1 = pd.read_csv(r'C:\\Users\\nr00756\\OneDrive - University of Surrey\\diss\\structured_data.csv')

### Reseting display options ###
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')

### Explore Dataset ###
print("First 10 rows:")
print(d1.head(10))

print("\nLast 10 rows:")
print(d1.tail(10))

print("Columns in original data:")
print(list(d1.columns))

print("\n Displaying number of null values in each column:")
print(d1.isna().sum())

print("\n Number of Duplicates in the dataset:")
print(d1.duplicated().sum())

print("\n Shape of the dataset:")
print(d1.shape)
d1.info()

### Target Distribution ###
print("\nTarget Variable Distribution:")
print(d1['Target'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Target', data=d1, palette='Set2')
plt.title('Distribution of Target Variable')
plt.show()

### Set diisplay percision to show only 3 decimal places ###
pd.set_option('display.precision', 3)

### Summary Statistics ###
print("\n Summary Statistics:")
print(d1.describe()) 

### Correlation Matrix ###
corr_matrix = d1.select_dtypes(include='number').corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, fmt='.3f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()
print(corr_matrix)

### Box Plot for All Numerical Variables by Target ###

numerical_cols = d1.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Generate box plots for all numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Target', y=col, data=d1, palette='Set3')
    plt.title(f'{col} by Target Category')
    plt.show()
### Plot Histogrms for all numerical columns ###
sns.set(style='darkgrid')
d1.hist(bins=10, figsize=(40, 35), grid=True)
plt.show()

### Explore Variables ###
print("\n Unique Values of Marital Status:")
print(d1['Marital status'].unique())

print("\n Unique Values of Nationality:")
print(d1['Nationality'].unique())

print("\n Value counts of Nationality:")
print(d1.loc[:, 'Nationality'].value_counts())

print("\n Unique Values of Course:")
print(d1['Course'].unique())

### Pie Chart for Gender ###
gender_counts = d1['Gender'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()

### Map and Filter Target variable
target_mapping = {'Graduate': 1, 'Dropout': 0}
d2 = d1.copy()
d2 = d2[d2['Target'].isin(target_mapping.keys())]
d2['Target'] = d2['Target'].map(target_mapping)

### One-Hot Encoding categorical columns ###
categorical_cols = ['Marital status', 'Nationality', 'Course']
d2 = pd.get_dummies(d2, columns=categorical_cols, drop_first=True)

### Identifying necessary numerical columns ###
numerical_cols = d1.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != 'Target' and col not in categorical_cols]
print("Numerical columns:")
print(numerical_cols)

### Split Dataset into Features and Target ###
X = d2.drop(columns=['Target'])
y = d2['Target']

### Train-Test Split ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

### Scaling numerical features ###
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

#----------------------------------------#

### Logistic Regression Model ###
lr = LogisticRegression(random_state=seed)
lr.fit(X_train_scaled, y_train)

### 5-fold cross-validation for Logistic Regression ###
lr_cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)

### Random Forest Model ###
rf = RandomForestClassifier(n_estimators=100, random_state=seed)
rf.fit(X_train_scaled, y_train)

### 5-fold cross-validation for Random Forest ###
rf_cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)

### Logistic Regression cross-validation results ###
lr_cv_mean = lr_cv_scores.mean()
lr_cv_std = lr_cv_scores.std()
print(f"Logistic Regression - Mean CV Accuracy: {lr_cv_mean:.3f}, Standard Deviation: {lr_cv_std:.3f}")

### Random Forest cross-validation results ###
rf_cv_mean = rf_cv_scores.mean()
rf_cv_std = rf_cv_scores.std()
print(f"Random Forest - Mean CV Accuracy: {rf_cv_mean:.3f}, Standard Deviation: {rf_cv_std:.3f}")

### Predict on test set using Logistic Regression ###
y_pred_lr = lr.predict(X_test_scaled)

### Predict on test set using Random Forest ###
y_pred_rf = rf.predict(X_test_scaled)

### Evaluate Models on test set ###
print("\nLogistic Regression Test Performance:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
ConfusionMatrixDisplay.from_estimator(lr, X_test_scaled, y_test, cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

print("\nRandom Forest Test Performance:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
ConfusionMatrixDisplay.from_estimator(rf, X_test_scaled, y_test, cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.show()

### Calculate, Print and Plot ROC-AUC for all models ###
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"Logistic Regression AUC: {lr_auc:.3f}")
print(f"Random Forest AUC: {rf_auc:.3f}")

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr.predict_proba(X_test_scaled)[:, 1])
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf.predict_proba(X_test_scaled)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', color='blue')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', color='green')

plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

### Feature Importance for Random Forest Model ###
importances = rf.feature_importances_
feature_names = X_train.columns

fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance for Random Forest Model:")
print(fi_df)

#----------------------------------------------------#

### Identify categorical feature names and indices ###
categorical_feature_names = [col for col in X_train.columns if col not in numerical_cols]
categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_feature_names]

print("Categorical feature names:")
print(categorical_feature_names)
print("Categorical feature indices:")
print(categorical_feature_indices)

### LIME setup for tabular data using unscaled data ###
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Dropout', 'Graduate'],
    mode='classification',
    categorical_features=categorical_feature_indices
)

### Define a prediction function that scales numerical data before predicting ###
def predict_fn_unscaled(data):
    # Convert data to DataFrame
    data_df = pd.DataFrame(data, columns=X_train.columns)
    # Scale numerical columns
    data_df[numerical_cols] = scaler.transform(data_df[numerical_cols])
    # Return prediction probabilities
    return rf.predict_proba(data_df)

### LIME explanation for test instance 0 in Random Forest Model###
i = 0  

exp_rf = explainer.explain_instance(
    data_row=X_test.iloc[i].values,
    predict_fn=predict_fn_unscaled
)

print("\nExplanation for Instance", i)
print(exp_rf.as_list()) 

### LIME explanation for test instance 1 in Random Forest Model###
i = 1

exp_rf = explainer.explain_instance(
    data_row=X_test.iloc[i].values,
    predict_fn=predict_fn_unscaled
)

print("\nExplanation for Instance", i)
print(exp_rf.as_list()) 

### LIME explanation for test instance 2 in Random Forest Model###
i = 2 

exp_rf = explainer.explain_instance(
    data_row=X_test.iloc[i].values,
    predict_fn=predict_fn_unscaled
)

print("\nExplanation for Instance", i)
print(exp_rf.as_list()) 

### Identify correctly classified instances ###
correctly_classified_indices = np.where(y_pred_rf == y_test)[0]
print(f"\nNumber of Correctly Classified Instances: {len(correctly_classified_indices)}")
print(f"Indices of Correctly Classified Instances: {correctly_classified_indices}")

### Apply LIME to a few correctly classified instances ###
num_instances_to_explain = min(3, len(correctly_classified_indices))  
for i in correctly_classified_indices[:num_instances_to_explain]:
    print(f"\nExplaining Correctly Classified Instance {i}:")
    exp_rf = explainer.explain_instance(
        data_row=X_test.iloc[i].values,  
        predict_fn=predict_fn_unscaled  
    )
    print(exp_rf.as_list())

### Identify misclassified instances ###
misclassified_indices = np.where(y_pred_rf != y_test)[0]
print(f"\nNumber of Misclassified Instances: {len(misclassified_indices)}")
print(f"Indices of Misclassified Instances: {misclassified_indices}")

#### Apply LIME to a few misclassified instances ###
num_instances_to_explain = min(3, len(misclassified_indices))  
for i in misclassified_indices[:num_instances_to_explain]:
    print(f"\nExplaining Misclassified Instance {i}:")
    exp_rf = explainer.explain_instance(
        data_row=X_test.iloc[i].values,  
        predict_fn=predict_fn_unscaled  
    )
    print(exp_rf.as_list())
