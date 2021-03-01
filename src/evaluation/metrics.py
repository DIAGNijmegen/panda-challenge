"""
Metric functions that can evaluate a submission.

Each function has as input:
- A reference DataFrame with the ground truth
- A submission DataFrame with the team's predictions.

Each functions outputs a dictionary with one or more metrics.
"""

from sklearn import metrics as skmetrics

def count(reference, submission):
    """Simply return the length of the dataset"""

    return {
        'N': len(reference),
        'N_tumor': len(reference[reference.isup_grade > 0])
    }

def qwk(reference, submission):
    """Quadratically weighted Cohen's kappa"""
    return {'qwk': skmetrics.cohen_kappa_score(
        y1=reference.isup_grade,
        y2=submission.isup_grade,
        labels=[0,1,2,3,4,5],
        weights='quadratic'
    )}

def lwk(reference, submission):
    """Linear weighted Cohen's kappa"""

    return {'lwk': skmetrics.cohen_kappa_score(
        y1=reference.isup_grade,
        y2=submission.isup_grade,
        labels=[0,1,2,3,4,5],
        weights='linear'
    )}

def acc(reference, submission):
    """Accuracy"""

    return {'acc': skmetrics.accuracy_score(
        y_true=reference.isup_grade,
        y_pred=submission.isup_grade,
    )}

def acc_tumor_only(reference, submission):
    """Accuracy on tumor cases"""

    reference_tumor = reference[reference.isup_grade > 0]
    submission_tumor = submission[reference.isup_grade > 0]

    return {'acc_gg_tumor': skmetrics.accuracy_score(
        y_true=reference_tumor.isup_grade,
        y_pred=submission_tumor.isup_grade,
    )}


def screening_tumor(reference, submission):
    """Compute screening metrics (e.g. sensitivity) for tumor vs benign"""

    cm = skmetrics.confusion_matrix(y_true=reference.isup_grade > 0,
                                    y_pred=submission.isup_grade > 0,
                                    normalize=None,
                                    labels=[0,1],
    )
    tn, fp, fn, tp = cm.ravel()

    # For source of these calculations, check:
    # https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    return {
        'acc_tumor':            ((tp + tn) / (tp + tn + fp + fn)),
        'f1_tumor':             ((2 * tp) / ((2 * tp) + fp + fn)),
        'sensitivity_tumor':    (tp / (tp + fn)), #a.k.a recall
        'specificity_tumor':    (tn / (tn + fp)),
        'precision_tumor':      (tp / (tp + fp)), # a.k.a positive predictive value
        'npv_tumor':            (tn / (tn + fn)), # negative predictive value
        'fnr_tumor':            (fn / (fn + tp)), # a.k.a miss rate
    }

def screening_gg2(reference, submission):
    """Compute screening metrics (e.g. sensitivity) for >= gg2"""

    cm = skmetrics.confusion_matrix(y_true=reference.isup_grade > 1,
                                    y_pred=submission.isup_grade > 1,
                                    normalize=None,
                                    labels=[0,1],
    )
    tn, fp, fn, tp = cm.ravel()

    # For source of these calculations, check:
    # https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    return {
        'acc_gg2':            ((tp + tn) / (tp + tn + fp + fn)),
        'f1_gg2':             ((2 * tp) / ((2 * tp) + fp + fn)),
        'sensitivity_gg2':    (tp / (tp + fn)), #a.k.a recall
        'specificity_gg2':    (tn / (tn + fp)),
        'precision_gg2':      (tp / (tp + fp)), # a.k.a positive predictive value
        'npv_gg2':            (tn / (tn + fn)), # negative predictive value
        'fnr_gg2':            (fn / (fn + tp)), # a.k.a miss rate
    }

def screening_gg3(reference, submission):
    """Compute screening metrics (e.g. sensitivity) for >= gg3"""

    cm = skmetrics.confusion_matrix(y_true=reference.isup_grade > 2,
                                    y_pred=submission.isup_grade > 2,
                                    normalize=None,
                                    labels=[0,1],
    )
    tn, fp, fn, tp = cm.ravel()

    # For source of these calculations, check:
    # https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    return {
        'acc_gg3':            ((tp + tn) / (tp + tn + fp + fn)),
        'f1_gg3':             ((2 * tp) / ((2 * tp) + fp + fn)),
        'sensitivity_gg3':    (tp / (tp + fn)), #a.k.a recall
        'specificity_gg3':    (tn / (tn + fp)),
        'precision_gg3':      (tp / (tp + fp)), # a.k.a positive predictive value
        'npv_gg3':            (tn / (tn + fn)), # negative predictive value
        'fnr_gg3':            (fn / (fn + tp)), # a.k.a miss rate
    }
