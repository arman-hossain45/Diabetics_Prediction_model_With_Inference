import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import  SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('diabetics.csv')



# Target and features
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# =====================
# Column split
# =====================
numeric_feature=X.select_dtypes(include=['int64','float64']).columns

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer,numeric_feature),
    
])

# =====================
# Random Forest Model
# =====================
rf_model = RandomForestClassifier(
   max_depth=5,
   max_features='log2',
   min_samples_leaf=2,
   min_samples_split=2,
   n_estimators=50
)

# =====================
# Full Pipeline
# =====================
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# =====================
# Train-test split
# ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = rf_pipeline.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)

print(" accuracy is :",accuracy)
metrix=confusion_matrix(y_test,y_pred)
print("confusion metrix is ",metrix)


# =====================
# Save model (IMPORTANT)
# =====================

with open("diabetics.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("Random Forest pipeline saved as diabetics_rf_pipeline.pkl")