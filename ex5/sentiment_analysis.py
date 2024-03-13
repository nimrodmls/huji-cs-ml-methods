import torch
import torchtext
import spacy
import numpy as np
import matplotlib.pyplot as plt
from torchtext.data import get_tokenizer
from torch.utils.data import random_split
from torchtext.experimental.datasets import IMDB
from torch.utils.data import DataLoader
from models import MyTransformer
from tqdm import tqdm
import torch.nn.functional as F
import os

def pad_trim(data):
    ''' Pads or trims the batch of input data.

    Arguments:
        data (torch.Tensor): input batch
    Returns:
        new_input (torch.Tensor): padded/trimmed input
        labels (torch.Tensor): batch of output target labels
    '''
    data = list(zip(*data))
    # Extract target output labels
    labels = torch.tensor(data[0]).float().to(device)
    # Extract input data
    inputs = data[1]

    # Extract only the part of the input up to the MAX_SEQ_LEN point
    # if input sample contains more than MAX_SEQ_LEN. If not then
    # select entire sample and append <pad_id> until the length of the
    # sequence is MAX_SEQ_LEN
    new_input = torch.stack([torch.cat((input[:MAX_SEQ_LEN],
                                        torch.tensor([pad_id] * max(0, MAX_SEQ_LEN - len(input))).long()))
                             for input in inputs])

    return new_input, labels

def split_train_val(train_set):
    ''' Splits the given set into train and validation sets WRT split ratio
    Arguments:
        train_set: set to split
    Returns:
        train_set: train dataset
        valid_set: validation dataset
    '''
    train_num = int(SPLIT_RATIO * len(train_set))
    valid_num = len(train_set) - train_num
    generator = torch.Generator().manual_seed(SEED)
    train_set, valid_set = random_split(train_set, lengths=[train_num, valid_num],
                                        generator=generator)
    return train_set, valid_set

def load_imdb_data(batch_size):
    """
    This function loads the IMDB dataset and creates train, validation and test sets.
    It should take around 15-20 minutes to run on the first time (it downloads the GloVe embeddings, IMDB dataset and extracts the vocab).
    Don't worry, it will be fast on the next runs. It is recommended to run this function before you start implementing the training logic.
    :return: train_set, valid_set, test_set, train_loader, valid_loader, test_loader, vocab, pad_id
    """
    cwd = os.getcwd()
    if not os.path.exists(cwd + '/.vector_cache'):
        os.makedirs(cwd + '/.vector_cache')
    if not os.path.exists(cwd + '/.data'):
        os.makedirs(cwd + '/.data')
    # Extract the initial vocab from the IMDB dataset
    vocab = IMDB(data_select='train')[0].get_vocab()
    # Create GloVe embeddings based on original vocab word frequencies
    glove_vocab = torchtext.vocab.Vocab(counter=vocab.freqs,
                                        max_size=MAX_VOCAB_SIZE,
                                        min_freq=MIN_FREQ,
                                        vectors=torchtext.vocab.GloVe(name='6B'))
    # Acquire 'Spacy' tokenizer for the vocab words
    tokenizer = get_tokenizer('spacy', 'en_core_web_sm')
    # Acquire train and test IMDB sets with previously created GloVe vocab and 'Spacy' tokenizer
    train_set, test_set = IMDB(tokenizer=tokenizer, vocab=glove_vocab)
    vocab = train_set.get_vocab()  # Extract the vocab of the acquired train set
    pad_id = vocab['<pad>']  # Extract the token used for padding

    train_set, valid_set = split_train_val(train_set)  # Split the train set into train and validation sets

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=pad_trim)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=pad_trim)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=pad_trim)
    return train_set, valid_set, test_set, train_loader, valid_loader, test_loader, vocab, pad_id

def run_training_epoch(model, criterion, optimizer, train_loader, validation_loader, device):
    """
    Run a single training epoch, recording loss and accuracy for both training and validation sets.
    Iterations on the validation set are evaluation only, no backpropagation is performed.

    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    train_ep_loss = 0
    for (features, labels) in tqdm(train_loader, total=len(train_loader)):
        model.train()
        # perform a training iteration
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()

    val_ep_loss = 0
    correct = 0
    for (features, labels) in tqdm(validation_loader, total=len(validation_loader)):
        # perform an evaluation
        with torch.no_grad():
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.float())

            predicted = (outputs.data > 0).float()
            correct += (predicted.squeeze() == labels.float()).sum().item()
            val_ep_loss += loss.item()

    return train_ep_loss / len(train_loader), \
           val_ep_loss / len(validation_loader), \
           correct / len(validation_loader.dataset)

### EXPERIMENTATION CODE ###

# VOCAB AND DATASET HYPERPARAMETERS, DO NOT CHANGE
MAX_VOCAB_SIZE = 25000 # Maximum number of words in the vocabulary
MIN_FREQ = 10 # We include only words which occur in the corpus with some minimal frequency
MAX_SEQ_LEN = 500 # We trim/pad each sentence to this number of words
SPLIT_RATIO = 0.8 # Split ratio between train and validation set
SEED = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)
torch.manual_seed(42)

batch_size = 32
num_of_blocks = 1
num_of_epochs = 5
learning_rate = 0.0001

# Load the IMDB dataset
train_set, valid_set, test_set, train_loader, valid_loader, test_loader, vocab, pad_id = load_imdb_data(batch_size)

model = MyTransformer(vocab=vocab, max_len=MAX_SEQ_LEN, num_of_blocks=num_of_blocks).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Using the BCEWithLogitsLoss as the criterion since we require binary classification (positive/negative)
criterion = torch.nn.BCEWithLogitsLoss()

# Training the model
train_losses = []
validation_losses = []
validation_accuracies = []
for num_epoch in range(num_of_epochs):
    print(f'Training epoch {num_epoch + 1}...')
    train_loss, val_loss, val_acc = run_training_epoch(
        model, criterion, optimizer, train_loader, valid_loader, device)

    train_losses.append(train_loss)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_acc)

# Print the final accuracy - on the validation set
print(f'Validation Accuracy: {validation_accuracies[-1]}')

# Plot the training and validation losses
plt.figure()
plt.plot(train_losses, label='Training loss', color='blue')
plt.plot(validation_losses, label='Validation loss', color='green')
plt.title('Transformer Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

