import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, scaling):
        super(ScaledDotProductAttention, self).__init__()

        self.scaling = scaling

    def forward(self, queries, keys, values, mask=None):
        x = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        x /= self.scaling
        if mask is not None:
            x
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, values)
        return x

# TODO add masking

class MultiHeadAttention(nn.Module):

    def __init__(self, scaling, num_heads, head_dim, model_dim):
        super(MultiHeadAttention, self).__init__()

        self.attention = ScaledDotProductAttention(scaling)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim

        self.query_linear = nn.Linear(
            model_dim, self.num_heads * self.head_dim)
        
        self.key_linear = nn.Linear(
            model_dim, self.num_heads * self.head_dim)
        
        self.values_linear = nn.Linear(
            model_dim, self.num_heads * self.head_dim)

        self.final_linear = nn.Linear(
            self.num_heads * self.head_dim, model_dim)

    def forward(self, queries, keys, values):
        """[batch_size, length, model_dim]"""
        batch_size, n_queries, model_dim = queries.shape
        batch_size, n_keys, model_dim = keys.shape
        batch_size, n_values, model_dim = values.shape

        queries = self.query_linear(queries).view(
            batch_size, n_queries, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3)
        
        keys = self.key_linear(keys).view(
            batch_size, n_keys, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3)
            
        values = self.key_linear(values).view(
            batch_size, n_values, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3)

        x = self.attention(queries, keys, values)

        x = x.permute(0, 2, 1, 3).contiguous().view(
            batch_size, n_queries, model_dim)
        
        x = self.final_linear(x)

        return x


class FeedForward(nn.Module):

    def __init__(self):
        super(FeedForward, self).__init__()


class EncoderLayer(nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()


class DecoderLayer(nn.Module):

    def __init__(self):
        super(DecoderLayer, self).__init__()


class EncoderStack(nn.Module):

    def __init__(self):
        super(EncoderStack, self).__init__()


class MaskCreator(nn.Module):

    def __init__(self):
        super(MaskCreator, self).__init__()


class DecoderStack(nn.Module):

    def __init__(self):
        super(DecoderStack, self).__init__()


class Embeddings(nn.Module):

    def __init__(self):
        super(Embeddings, self).__init__()


class PositionalEncoding(nn.Module):

    def __init__(self):
        super(PositionalEncoding, self).__init__()


class DecoderOutputHandler(nn.Module):

    def __init__(self):
        super(DecoderOutputHandler, self).__init__()


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, enc_inp, dec_inp):
        pass






