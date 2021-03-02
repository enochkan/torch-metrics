from collections import namedtuple

import torch

torch.manual_seed(42)

Input = namedtuple("Input", ["preds", "target"])
NUM_BATCHES = 10
BATCH_SIZE = 16
NUM_CLASSES = 10
binary_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)


binary_raw_inputs = Input(
    preds=torch.randint(
        high=2,
        size=(
            NUM_BATCHES,
            BATCH_SIZE,
        ),
    ),
    target=torch.randint(
        high=2,
        size=(
            NUM_BATCHES,
            BATCH_SIZE,
        ),
    ),
)


multiclass_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)


multiclass_inputs = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)

uniform_regression_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE) * 10,
    target=torch.rand(NUM_BATCHES, BATCH_SIZE) * 10,
)
normal_regression_inputs = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE) * 10,
    target=torch.randn(NUM_BATCHES, BATCH_SIZE) * 10,
)
