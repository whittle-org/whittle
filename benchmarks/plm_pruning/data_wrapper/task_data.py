
GLUE_TASK_INFO = {
    "cola": {
        "metric": "matthews_correlation",
        "mode": "max",
        "seq_length": 64,
        "keys": ("sentence", None),
    },
    "mnli": {
        "metric": "accuracy",
        "mode": "max",
        "seq_length": 128,
        "keys": ("premise", "hypothesis"),
    },
    "mrpc": {
        "metric": "f1",
        "mode": "max",
        "seq_length": 128,
        "keys": ("sentence1", "sentence2"),
    },
    "qnli": {
        "metric": "accuracy",
        "mode": "max",
        "seq_length": 128,
        "keys": ("question", "sentence"),
    },
    "qqp": {
        "metric": "f1",
        "mode": "max",
        "seq_length": 128,
        "keys": ("question1", "question2"),
    },
    "rte": {
        "metric": "accuracy",
        "mode": "max",
        "seq_length": 128,
        "keys": ("sentence1", "sentence2"),
    },
    "sst2": {
        "metric": "accuracy",
        "mode": "max",
        "seq_length": 64,
        "keys": ("sentence", None),
    },
    "stsb": {
        "metric": "spearmanr",
        "mode": "max",
        "seq_length": 128,
        "keys": ("sentence1", "sentence2"),
    },
}
