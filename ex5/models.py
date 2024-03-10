import torch
import torch.nn as nn
import torch.nn.functional as F

class MySelfAttention(nn.Module):
    """
    Self attention layer
    """
    def __init__(self, input_dim):
        """
        :param input_dim: The embedding (feature) dimension the input tokens (d).
        """
        super(MySelfAttention, self).__init__()
        self.input_dim = input_dim
        # The linear layers for the query, key and value matrices.
        # These are of dimensions d x d.
        self.w_query = nn.Linear(self.input_dim, self.input_dim)
        self.w_key = nn.Linear(self.input_dim, self.input_dim)
        self.w_value = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x):
        """
        :param x: The input tensor of shape (B, T, d).
                  Where T is the number of tokens and d is the 
                  embedding (feature) dimension of each token,
                  and B is the batch size.
        """
        # Computing the Q, K, V matrices by the corresponding linear layers to the input
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)

        # Transpose each matrix in K, the number of matrices depend on the batch size
        # done along the axis 1 and 2: (T, d) -> (d, T)
        k_t = torch.transpose(k, 1, 2)

        # Normalizing and computing the attention score
        normalized = torch.softmax(torch.bmm(q, k_t) / torch.sqrt(torch.tensor(self.input_dim)), dim=-1)
        return torch.bmm(normalized, v)

class MyLayerNorm(nn.Module):
    """
    Layer Normalization layer.
    """
    def __init__(self, input_dim):
        """
        :param input_dim: The dimension of the input (T, d).
        """
        super(MyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(*input_dim))
        self.beta = nn.Parameter(torch.zeros(*input_dim))
        self.eps = 1e-8

    def forward(self, x):
        """
        :param x: The input tensor of shape (T, d).
        """
        # Compute the mean and variance for every element in the batch
        mean = x.mean(dim=(1, 2), keepdim=True)
        # Not taking regard of the correction, computing the variance as `
        # described in the assignment
        variance = x.var(dim=(1, 2), correction=0, keepdim=True)
        #variance2 = ((x - mean) ** 2).mean(dim=(1,2), keepdim=True)
        return (self.gamma * ((x - mean) / torch.sqrt(variance + self.eps))) + self.beta

class MyTransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, max_len, input_dim):
        super(MyTransformerBlock, self).__init__()
        self.attention = MySelfAttention(input_dim)
        self.norm1 = MyLayerNorm((max_len, input_dim))
        self.norm2 = MyLayerNorm((max_len, input_dim))
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.attention(x)
        x = self.norm1(self.dropout(out) + x)
        out = self.fc2(F.relu(self.fc1(x)))
        out = self.norm2(out + x)
        return out

class MyTransformer(nn.Module):
    """
    Transformer.
    """
    def __init__(self, vocab, max_len, num_of_blocks):
        """
        :param vocab: The vocabulary object.
        :param num_of_blocks: The number of transformer blocks.
        """
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.emb_dim = self.embedding.embedding_dim
        self.max_len = max_len
        self.blocks = nn.ModuleList([MyTransformerBlock(self.max_len, self.emb_dim) for _ in range(num_of_blocks)])
        self.fc = nn.Linear(self.emb_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        avg_pooling = x.mean(dim=1)
        x = self.fc(avg_pooling)
        return x

