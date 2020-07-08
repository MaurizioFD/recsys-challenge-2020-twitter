import numpy as np
from torch.utils.data import DataLoader, SequentialSampler

HIDDEN_SIZE_BERT = 768

def flat_accuracy(preds, labels):
    preds = preds.squeeze()
    my_round = lambda x: 1 if x >= 0.5 else 0
    pred_flat = np.fromiter(map(my_round, preds), dtype=np.int).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_max_len(sentences):
    max_len = 0
    # For every sentence...
    for sent in sentences:
        # Update the maximum sentence length.
        max_len = max(max_len, len(sent))

    #print('Max sentence length: ', max_len)
    return max_len


def get_max_len_cap(sentences, cap: int = 128) -> (int, bool):
    is_capped = False

    max_len = 0
    # For every sentence...
    for sent in sentences:
        # Update the maximum sentence length.
        max_len = max(max_len, len(sent))
        # check if the value is higher than the cap
        if max_len >= cap:
            is_capped = True
            max_len = cap
            break

    #print('Max sentence length: ', max_len)
    #print('Is capped: ', is_capped)
    return max_len, is_capped


def create_data_loaders(train_dataset, val_dataset, batch_size=3):
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=SequentialSampler(train_dataset),  # Select batches sequentially
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return train_dataloader, validation_dataloader


def format_time(elapsed):
    import datetime
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
