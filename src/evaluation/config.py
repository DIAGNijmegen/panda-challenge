"""
Common config variables shared across scripts
"""
import string

import evaluation.metrics

# Path template to submission files.
SUBMISSION_PATH = "{base_dir}/{team}/{dataset}/{run}/{submission}.csv"

# Metric functions to run
BOOTSTRAPPED_METRICS = [
    evaluation.metrics.qwk,
    evaluation.metrics.lwk,
    evaluation.metrics.acc,
    evaluation.metrics.acc_tumor_only,
    evaluation.metrics.screening_tumor,
    evaluation.metrics.screening_gg2,
    evaluation.metrics.screening_gg3,
]
METRICS = [
    evaluation.metrics.count,
    evaluation.metrics.qwk,
    evaluation.metrics.lwk,
    evaluation.metrics.acc,
    evaluation.metrics.acc_tumor_only,
    evaluation.metrics.screening_tumor,
    evaluation.metrics.screening_gg2,
    evaluation.metrics.screening_gg3,
]

# Datasets used in the analysis
DATASETS = {
    'example': {
        'reference': 'reference.csv',
        'dir': 'example',
        'usage': 'Public',
        'friendly_name': "Example data set",
        'image_ids': None,
        'generate_confusion_matrix': False,
    },
}

# Teams that are evaluated
TEAMS = {
   "example":{
      "friendly_name": "Example team",
      "identifier": "example"
   },
}
