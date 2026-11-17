# Football Player Position Classification

University machine learning lab (AUCA, Lab 4). Python, pandas, scikit-learn, matplotlib.

## Task
Predict a Premier League player's position from 12 defensive statistics
(tackles, pressures, blocks, interceptions, clearances) and compare
a decision tree with k-NN.

## Part I: Multi-class classification
- Position labels include combined classes (e.g. FW,MF), giving 10 classes with strong
  imbalance: DF is ~35% of the data, FW,DF and DF,FW are ~1% each.
- Stratified 70/15/15 train/validation/test split, features standardized (StandardScaler).
- Decision tree tuned with GridSearchCV (5-fold, macro F1) over criterion, max_depth,
  min_samples_split and min_samples_leaf. Best: gini, no depth limit,
  min_samples_leaf=4, min_samples_split=10.
- k-NN with k=5 (default value, not tuned).
- Both models compared with 5-fold cross-validation on macro F1.

| Model | Mean macro F1 (5-fold CV) |
|-------|---------------------------|
| k-NN (k=5) | 0.322 |
| Decision tree (tuned) | 0.345 |

Decision tree on the test set (72 players): accuracy 0.67, macro F1 0.40.
Frequent classes are predicted well (DF F1 0.91, GK 0.80), while rare combined
labels (FW,MF, MF,FW, MF,DF, FW,DF) get F1 = 0.

## Part II: Binary classification (FW vs MF)
- Kept only players labeled FW or MF: 177 players (100 MF, 77 FW).
- Stratified 75/25 split (45 test players), decision tree with max_depth=4.
- ROC curve: AUC = 0.75.
- Evaluated thresholds 0.3, 0.5 and 0.7 on the predicted FW probability:

| Threshold | FW precision | FW recall | FW F1 | Accuracy |
|-----------|--------------|-----------|-------|----------|
| 0.3 | 0.61 | 0.85 | 0.71 | 0.69 |
| 0.5 | 0.65 | 0.65 | 0.65 | 0.69 |
| 0.7 | 0.65 | 0.65 | 0.65 | 0.69 |

Lowering the threshold to 0.3 finds more forwards (recall 0.85) at the cost of more
false positives (precision 0.61), so it fits a task where missing a forward is worse
than a false alarm. Thresholds 0.5 and 0.7 give identical results because a shallow
decision tree outputs only a few distinct probability values.

## Observations and limitations
- The difference between the models (0.345 vs 0.322) is small compared to the spread
  across folds, so it is not a clear win. Also, the tree was tuned while k-NN was not.
- Combined labels with very few examples are the main source of errors; merging them
  would likely help.
- Some classes have fewer members than the number of CV folds, so CV estimates are noisy.
- The scaler is fitted on the whole train+val set before cross-validation, which causes
  mild data leakage; a scikit-learn Pipeline would fix this.

## How to run
pip install numpy pandas scikit-learn matplotlib
python main.py

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib
