
from sklearn.metrics import f1_score, classification_report

def get_macro_f1_score(y_preds, y_trues):
    return f1_score(y_trues, y_preds, average='macro', zero_division=0)

def get_micro_f1_score(y_preds, y_trues):
    return f1_score(y_true=y_trues, y_pred=y_preds, average='micro')

def get_weighted_f1_score(y_preds, y_trues):
    return f1_score(y_true=y_trues, y_pred=y_preds, average='weighted')


def get_classification_report(y_preds, y_trues):
    report = classification_report(y_trues, y_preds)
    return report


