import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc

# ==========================================
# 0. LOAD AND PREPROCESS
# ==========================================
df = pd.read_csv("PL_Players.csv", encoding="latin-1").dropna()

df = df.rename(columns={
    "Pos": "Position", "90s": "Total minutes played divided by 90", "Tkl": "Tackles",
    "TklW": "Tackles Won", "Def 3rd": "Tackles won in defensive 3rd", "Mid 3rd": "Tackles won in midfield",
    "Att 3rd": "Tackles won in attacking 3rd", "Press": "number of times opponent put under pressure",
    "Succ": "Successful press", "%": "% of successful press", "Def 3rd.1": "Press in defense",
    "Mid 3rd.1": "Press in midfield", "Att 3rd.1": "Press in attack", "Blocks": "Blocked the ball",
    "Pass": "Passes blocked", "Int": "Interceptions", "Tkl+Int": "Tackles+Interceptions",
    "Clr": "Clearances", "Err": "Errors leading to goal"
})

features = ["Tackles", "Tackles Won", "Tackles won in defensive 3rd", "Tackles won in midfield",
            "Tackles won in attacking 3rd", "Press in defense", "Press in midfield", "Press in attack",
            "Blocked the ball", "Passes blocked", "Interceptions", "Clearances"]

X = df[features]
le = LabelEncoder()
y = le.fit_transform(df["Position"])

# ==========================================
# 1. SPLIT DATASET (Train, Test, Validation)
# ==========================================
# Using an 70 / 15 / 15 ratio
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val)

# Scale features based on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_train_val_scaled = scaler.fit_transform(X_train_val) # For full Cross-Validation later

# ==========================================
# 2. SELECT RELEVANT FEATURES
# ==========================================
# Features are preserved from Lab 3 configuration as assigned.

# ==========================================
# 3 & 4. DECISION TREE & HYPERPARAMETER TUNING
# ==========================================
dt_base = DecisionTreeClassifier(random_state=42)

# Define parameter options to fine-tune tree depth and node splitting rules
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Tune using GridSearch on our split training/validation setup
grid_search = GridSearchCV(dt_base, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_scaled, y_train)

best_dt = grid_search.best_estimator_
print(f"Optimal Hyperparameters: {grid_search.best_params_}\n")

# ==========================================
# 5. ANALYZE CLASS BALANCE
# ==========================================
print("--- Class Distribution Analysis ---")
print(df["Position"].value_counts(normalize=True))
print("-" * 40)

# ==========================================
# 6. EVALUATE DESIGNATED METRICS
# ==========================================
y_pred_dt = best_dt.predict(X_test_scaled)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print("Classification Report:")
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred_dt), zero_division=0))

# 7. COMPARE KNN VS DECISION TREE USING F1-SCORE CROSS-VALIDATION
# ==========================================

knn_model = KNeighborsClassifier(n_neighbors=5)

# Perform 5-fold cross-validation using macro F1-score
knn_cv = cross_val_score(
    knn_model,
    X_train_val_scaled,
    y_train_val,
    cv=5,
    scoring='f1_macro'
)

dt_cv = cross_val_score(
    best_dt,
    X_train_val_scaled,
    y_train_val,
    cv=5,
    scoring='f1_macro'
)

print("\n--- KNN vs Decision Tree: F1-Score Cross-Validation ---")

print("\nKNN F1-scores for each fold:")
print(knn_cv)
print(f"KNN Mean F1-score: {knn_cv.mean():.4f}")

print("\nDecision Tree F1-scores for each fold:")
print(dt_cv)
print(f"Decision Tree Mean F1-score: {dt_cv.mean():.4f}")

print("\n--- Final Comparison ---")

if knn_cv.mean() > dt_cv.mean():
    print("KNN performs better based on the average F1-score.")
elif dt_cv.mean() > knn_cv.mean():
    print("Decision Tree performs better based on the average F1-score.")
else:
    print("Both models have the same average F1-score.")
# ==========================================
# PART II: 1. ISOLATE POSITION INSTANCES 'FW' & 'MF'
# ==========================================
binary_df = df[df['Position'].isin(['FW', 'MF'])].copy()

print("\n==============================")
print("PART II: BINARY CLASSIFICATION")
print("==============================")
print("Binary Target Balance Metrics:")
print(binary_df['Position'].value_counts())

# Construct absolute integer flags for the target index map
binary_df['Target'] = binary_df['Position'].map({'MF': 0, 'FW': 1})

X_bin = binary_df[features]
y_bin = binary_df['Target']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bin, y_bin, test_size=0.25, random_state=42, stratify=y_bin)

# Scale subset dimensions
bin_scaler = StandardScaler()
X_train_b_scaled = bin_scaler.fit_transform(X_train_b)
X_test_b_scaled = bin_scaler.transform(X_test_b)

# ==========================================
# PART II: 2. PERFORM RUNTIME CLASSIFICATION
# ==========================================
# Initializing using an optimized binary instance parameter configuration
binary_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
binary_clf.fit(X_train_b_scaled, y_train_b)

# Capture target continuous probability arrays
y_probs = binary_clf.predict_proba(X_test_b_scaled)[:, 1]

# ==========================================
# PART II: 3. PLOT AND INTERPRET ROC CURVE
# ==========================================
fpr, tpr, thresholds = roc_curve(y_test_b, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for FW vs MF Classification')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# PART II: 4. EVALUATE ALTERNATING THRESHOLD PLACEMENT
# ==========================================
print("\nTesting Model Predictions at Variant Confidence Thresholds:")
for t in [0.3, 0.5, 0.7]:
    preds_at_t = (y_probs >= t).astype(int)
    print(f"\n--- Threshold Boundary: {t} ---")
    print(classification_report(y_test_b, preds_at_t, target_names=['MF', 'FW'], zero_division=0))
