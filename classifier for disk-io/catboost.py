import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef

def sliding_window(data, window_size=3):
    return np.array([data.iloc[i:i + window_size].values.flatten()
                     for i in range(len(data) - window_size + 1)])

columns_to_keep = [0, 1, 2, 3, 5, 6] 

benign_train = pd.read_csv('../blk/benign-80.csv', usecols=columns_to_keep)
mal_train = pd.read_csv('../blk/mal-merged.csv', usecols=columns_to_keep)
benign_test = pd.read_csv('../blk/benign-20.csv', usecols=columns_to_keep)
unknown_test = pd.read_csv('../blk/unknown-merged.csv', usecols=columns_to_keep)

benign_train['label'] = 0
mal_train['label'] = 1
benign_test['label'] = 0
unknown_test['label'] = 1

X_benign_train = sliding_window(benign_train.drop('label', axis=1))
X_mal_train = sliding_window(mal_train.drop('label', axis=1))
X_benign_test = sliding_window(benign_test.drop('label', axis=1))
X_unknown_test = sliding_window(unknown_test.drop('label', axis=1))

X_train = np.concatenate([X_benign_train, X_mal_train])
y_train = np.concatenate([np.zeros(X_benign_train.shape[0]), np.ones(X_mal_train.shape[0])])
X_test = np.concatenate([X_benign_test, X_unknown_test])
y_test = np.concatenate([np.zeros(X_benign_test.shape[0]), np.ones(X_unknown_test.shape[0])])

sample_weights = np.where(y_train == 1, 8, 1)  
clf = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    l2_leaf_reg=1,
    bagging_temperature=0,
    random_strength=1,
    random_seed=42,
    verbose=True
)
clf.fit(X_train, y_train, sample_weight=sample_weights)

threshold = 0.5
y_test_pred = (clf.predict_proba(X_test)[:, 1] > threshold).astype(int)

def evaluate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  
    acc = (tp + tn) / (tp + tn + fp + fn)  
    mcc = matthews_corrcoef(y_true, y_pred)  
    return {'TNR': tnr, 'Recall': recall, 'F1-Score': f1, 'Accuracy': acc, 'Precision': precision, 'MCC': mcc}

test_metrics = evaluate_metrics(y_test, y_test_pred)
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

test_predictions_df = pd.DataFrame({'predicted_label': y_test_pred})
test_predictions_df.to_csv('test_predictions.csv', index=False)
