
import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn, optim
from torch.nn import functional

from tqdm import tqdm
from VAE import *

'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)'''


torch.manual_seed(42)   # args.seed


TRAIN_PATH = "tweet_tokens/embeddings/day_1/embeddings_day_1_clean_unique_COMPLETE.csv"
VALID_PATH = "tweet_tokens/embeddings/day_2/embeddings_day_2_clean_unique_COMPLETE.csv"

LOG_PATH = "vae_log.txt"

MODEL_PATH = "models/first_VAE.model"

DEVICE = "cpu"

LEARNING_RATE = 1e-3

BATCH_SIZE = 1024

N_EPOCHS = 1000

PATIENCE = 20

INPUT_DIM = 768
HIDDEN_DIM = 128
LATENT_DIM = 32

'''
def read_text_embeddings(_type, embeddings_file):
    if _type == "train":
        max_rows = 1024
    else:
        max_rows = 1024  # basically load all the rows
    text_embeddings = np.loadtxt(embeddings_file, delimiter=",", usecols=range(1,769), max_rows=max_rows, dtype=np.float32)  # 768 embeddings + 1 tweet_id = 769 columns
    
    return text_embeddings
'''


def train(model, train_set, batch_size, epoch):
    # set the train mode
    model.train()
    # loss of the epoch
    train_loss = 0
    
    train_set_length = len(train_set)
    
    iterator = range(0, train_set_length, batch_size)

    for batch_idx in tqdm(iterator, desc=f"\tTraining : "):

        batch_start = batch_idx
        batch_end = min(batch_start + batch_size, train_set_length)
        
        batch = train_set[batch_start:batch_end]
        batch = batch.float().to(DEVICE)
        
        # update the gradients to zero
        optimizer.zero_grad()
        # forward pass (batch_sample is decode(encode(batch)))
        batch_sample, z_mu, z_var = model(batch)
        # loss 
        loss = loss_function(batch_sample, batch, z_mu, z_var)
        # backward pass
        loss.backward()
        train_loss += loss.item()
        # update the weights
        optimizer.step()

    return train_loss


def test(model, test_set, batch_size, epoch):
    # set the evaluation mode
    model.eval()
    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        
        test_set_length = len(test_set)
    
        iterator = range(0, test_set_length, batch_size)

        for batch_idx in tqdm(iterator, desc=f"\tValidation : "):

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, test_set_length)

            batch = test_set[batch_start:batch_end]
            batch = batch.float().to(DEVICE)
            
            # reshape the data
            batch = batch.float().to(DEVICE)
            # forward pass
            batch_sample, z_mu, z_var = model(batch)
            # loss 
            loss = loss_function(batch_sample, batch, z_mu, z_var)
            test_loss += loss.item()

    return test_loss


encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
model = VAE(encoder, decoder).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


#model = torch.load(MODEL_PATH)


model = model.float()

best_valid_loss = float('inf')

epoch = 1
early_stopped = False


log_file = open(LOG_PATH, "w+")


while epoch <= N_EPOCHS and not early_stopped:
    
    print(f'\nEpoch {epoch} \n')
    
    train_loss = 0
    train_len = 0
    valid_loss = 0
    valid_len = 0
    
    # train 
    i = 0
    for chunk in pd.read_csv(TRAIN_PATH, header=0, usecols=range(1,769), chunksize=1000000):
        print("Epoch", epoch, "- Training on chunk : ", i)
        train_set = torch.tensor(chunk.values, dtype=torch.float32)
        print("Training set : ", train_set.size())
        train_loss += train(model, train_set, BATCH_SIZE, epoch)
        train_len += len(train_set)
        #del chunk
        i += 1
    
    print()
    # validate
    i = 0
    for chunk in pd.read_csv(VALID_PATH, header=0, usecols=range(1,769), chunksize=1000000):
        print("Epoch", epoch, "- Validating on chunk : ", i)
        valid_set = torch.tensor(chunk.values, dtype=torch.float32)
        print("Validation set : ", valid_set.size())
        valid_loss += test(model, valid_set, BATCH_SIZE, epoch)
        valid_len += len(valid_set)
        #del chunk
        i += 1

    train_loss /= train_len
    valid_loss /= valid_len

    print(f'\tTrain Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
    log_file.write(f'Epoch: {epoch} - Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f} \n')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        counter = 1
    else:
        counter += 1

    if counter > PATIENCE:
        early_stopped = True
        print(f"\nEarly stopped at epoch {epoch}")
        
    epoch += 1


torch.save(model, MODEL_PATH)
print("\nModel saved : ", MODEL_PATH)
log_file.write("\nModel saved : ", MODEL_PATH)

log_file.close()
