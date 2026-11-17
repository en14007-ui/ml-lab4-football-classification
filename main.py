import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve, auc

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
# 1. SPLIT: 85% train+validation (used inside 5-fold CV), 15% held-out test
# ==========================================
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ==========================================
# 2. TUNE BOTH MODELS ON THE SAME DATA WITH THE SAME CV
# ==========================================
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    cv=5, scoring='f1_macro'
)
dt_grid.fit(X_train_val, y_train_val)  # trees do not need scaling
best_dt = dt_grid.best_estimator_
print("Best tree params:", dt_grid.best_params_)

# k-NN: the scaler is inside the pipeline, so there is no leakage across folds
knn_grid = GridSearchCV(
    make_pipeline(StandardScaler(), KNeighborsClassifier()),
    {'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11, 15]},
    cv=5, scoring='f1_macro'
)
knn_grid.fit(X_train_val, y_train_val)
best_knn = knn_grid.best_estimator_
print("Best k:", knn_grid.best_params_)

# ==========================================
# 3. CLASS BALANCE
# ==========================================
print("\n--- Class Distribution Analysis ---")
print(df["Position"].value_counts(normalize=True))
print("-" * 40)

# ==========================================
# 4. TEST-SET EVALUATION OF THE DECISION TREE
# ==========================================
y_pred_dt = best_dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print("Classification Report:")
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred_dt), zero_division=0))

# ==========================================
# 5. KNN VS DECISION TREE
# ==========================================
print("\n--- KNN vs Decision Tree (both tuned, same data, 5-fold CV, macro F1) ---")
print(f"KNN CV macro F1:  {knn_grid.best_score_:.4f}")
print(f"Tree CV macro F1: {dt_grid.best_score_:.4f}")

print("\n--- Check on the held-out test set ---")
print(f"KNN test macro F1:  {f1_score(y_test, best_knn.predict(X_test), average='macro'):.4f}")
print(f"Tree test macro F1: {f1_score(y_test, y_pred_dt, average='macro'):.4f}")

# ==========================================
# PART II: BINARY CLASSIFICATION (FW vs MF)
# ==========================================
binary_df = df[df['Position'].isin(['FW', 'MF'])].copy()

print("\n==============================")
print("PART II: BINARY CLASSIFICATION")
print("==============================")
print("Binary Target Balance Metrics:")
print(binary_df['Position'].value_counts())

binary_df['Target'] = binary_df['Position'].map({'MF': 0, 'FW': 1})
X_bin = binary_df[features]
y_bin = binary_df['Target']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bin, y_bin, test_size=0.25, random_state=42, stratify=y_bin
)

binary_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
binary_clf.fit(X_train_b, y_train_b)
y_probs = binary_clf.predict_proba(X_test_b)[:, 1]

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
plt.savefig("roc_curve.png", dpi=150)
plt.show()

print("\nTesting Model Predictions at Variant Confidence Thresholds:")
for t in [0.3, 0.5, 0.7]:
    preds_at_t = (y_probs >= t).astype(int)
    print(f"\n--- Threshold Boundary: {t} ---")
    print(classification_report(y_test_b, preds_at_t, target_names=['MF', 'FW'], zero_division=0))