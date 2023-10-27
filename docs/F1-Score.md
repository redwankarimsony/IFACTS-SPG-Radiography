## F1-Score for Multiclass Classification

The F1-score is a measure of a test's accuracy, and it is defined as the harmonic mean of the precision and recall. For binary classification, the formula is straightforward:

$$ F1 = 2 \times \frac{precision \times recall}{precision + recall} $$

For multiclass classification, calculating the F1-score is a bit more involved since you have to consider each class. There are multiple ways to compute a multiclass F1-score:

### 1. Macro-average F1-score

Calculate the F1-score independently for each class and then take the average. This treats all classes equally, no matter their size.

$$ F1_{macro} = \frac{1}{N} \sum_{i=1}^{N} F1_i $$

where $N$ is the number of classes, and ${F1}_i$ is the F1 score for class $i$.

### 2. Micro-average F1-score

Aggregate the individual true positives, false positives, and false negatives for each class and then compute the F1-score on the aggregated numbers.

$$ F1_{micro} = 2 \times \frac{\sum TP}{\sum TP + \frac{1}{2}(\sum FP + \sum FN)} $$

where $TP$ is true positives, $FP$ is false positives, and $FN$ is false negatives summed over all classes.

### 3. Weighted F1-score

Compute the F1-score for each class independently but when averaging them, weigh them by the number of true instances for each class. This takes class imbalance into account.

$$ F1_{weighted} = \frac{1}{\sum w_i} \sum_{i=1}^{N} w_i \times F1_i $$

where $w_i$ is the number of true instances for class $i$, and $F1_i$ is the F1 score for class $i$.

To compute any of these, you'd typically need a confusion matrix for your multiclass classifier or, at the very least, the counts of true positives, false positives, and false negatives for each class.

In Python, `scikit-learn` library's `classification_report` or `f1_score` function can be utilized to calculate these metrics.

