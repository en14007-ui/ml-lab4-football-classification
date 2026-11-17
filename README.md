# Football Player Position Classification

University machine learning lab (AUCA, Lab 4). Python, pandas, scikit-learn, matplotlib.

## Task
Predict a Premier League player's position from 12 defensive statistics
(tackles, pressures, blocks, interceptions, clearances) and compare
a decision tree with k-NN.

## Part I: Multi-class classification
- Position labels include combined classes (e.g. FW,MF), giving 10 classes with strong
  imbalance: DF is ~35% of the data, FW,DF and DF,FW are ~1% each.
- Stratified split: 85% train+validation (5-fold cross-validation), 15% held-out test set (72 players).
- Both models tuned with GridSearchCV (5-fold, macro F1) on the same data:
  - Decision tree: criterion, max_depth, min_samples_split, min_samples_leaf
    (best: entropy, no depth limit, min_samples_leaf=1, min_samples_split=5)
  - k-NN: k in {3, 5, 7, 9, 11, 15} (best: k = 7); features are standardized inside
    a Pipeline, so there is no leakage between folds.

| Model | CV macro F1 | Test macro F1 |
|-------|-------------|---------------|
| k-NN (k=7) | 0.324 | 0.332 |
| Decision tree | 0.369 | 0.451 |

The decision tree is better on both estimates. Its test accuracy is 0.67.
Frequent classes are predicted well (DF F1 0.86, FW 0.82, GK 1.00), while rare
combined labels are mostly missed (MF,DF and FW,DF: F1 = 0; MF,FW: 0.33), which keeps
macro F1 low.

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
