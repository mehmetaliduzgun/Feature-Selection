# All required libraries are imported here.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Checking if there is any null values in dataframe.
print(crops.isna().sum())

# Checking the unique crop values.
print(crops['crop'].unique())

# Review the dataframe.
crops.head()

# Feature selection.
X = crops.drop('crop', axis=1)
y = crops['crop']

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

features_dict = {}
feature_performances_dict = {}
for feature in ['N', 'P', 'K', 'ph']:
    logreg = LogisticRegression(multi_class='multinomial')
    logreg.fit(X_train[[feature]], y_train)
    y_pred = logreg.predict(X_test[[feature]])
    feature_importance = metrics.f1_score(y_test,y_pred, average='weighted')
    feature_performances_dict[feature] = feature_importance
    print(f"F1-Score for {feature} : {feature_importance}")
best_feature = max(feature_performances_dict, key=feature_performances_dict.get)
best_predictive_feature = {best_feature: feature_performances_dict[best_feature]}
print("Best predictive feature:", best_predictive_feature)
