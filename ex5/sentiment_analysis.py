import torch
import torchtext
import spacy
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

def load_imdb_data():
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# VOCAB AND DATASET HYPERPARAMETERS, DO NOT CHANGE
MAX_VOCAB_SIZE = 25000 # Maximum number of words in the vocabulary
MIN_FREQ = 10 # We include only words which occur in the corpus with some minimal frequency
MAX_SEQ_LEN = 500 # We trim/pad each sentence to this number of words
SPLIT_RATIO = 0.8 # Split ratio between train and validation set
SEED = 0

# YOUR HYPERPARAMETERS
### YOUR CODE HERE ###
# batch_size = 32
# num_of_blocks =
# num_of_epochs =
# learning_rate =

# Load the IMDB dataset
train_set, valid_set, test_set, train_loader, valid_loader, test_loader, vocab, pad_id = load_imdb_data()

model = MyTransformer(vocab=vocab, max_len=MAX_SEQ_LEN, num_of_blocks=num_of_blocks).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### EXAMPLE CODE FOR RUNNING THE DATALOADER.
# for batch in tqdm(train_loader, desc='Train', total=len(train_loader)):
#     inputs_embeddings, labels = batch

### YOUR CODE HERE FOR THE SENTIMENT ANALYSIS TASK ###