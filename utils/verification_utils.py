import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def get_verification_pairs(file_names):
    """
    Args:
        file_names:

    Returns:

    """
    pairs = []
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            if file_names[i] == file_names[j]:
                # print(idx, file_names[i], file_names[j], 1)
                pairs.append((i, j, 1))
            else:
                # print(idx, file_names[i], file_names[j], 0)
                pairs.append((i, j, 0))

        # idx += 1
    return pairs


def getTMRatFMR(fpr, tpr, thresholds, target_fmr=0.01):
    # Find the threshold closest to your target FMR (1% in this case)
    closest_index = np.argmin(np.abs(fpr - target_fmr))
    closest_fmr = fpr[closest_index]
    closest_tmr = tpr[closest_index]
    closest_threshold = thresholds[closest_index]

    # print("Closest FMR:", closest_fmr)
    # print("Corresponding TMR:", closest_tmr)
    # print("Threshold at this point:", closest_threshold)

    print(f"TMR: {closest_tmr:0.4f} at FMR: {closest_fmr:0.4f} with threshold: {closest_threshold}")

    return closest_tmr, closest_threshold


def calculate_eer(y_trues, y_preds):
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer
