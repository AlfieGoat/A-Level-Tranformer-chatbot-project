import torch
from torch import nn
from torch.nn import functional as F
import math


class ScaledDotProductAttention(nn.Module):

    def __init__(self, scaling):
        super(ScaledDotProductAttention, self).__init__()

        self.scaling = scaling

    def forward(self, queries, keys, values, mask):
        x = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        x /= self.scaling
        if mask is not None:
            x += mask
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, values)
        return x

# TODO add masking


class MultiHeadAttention(nn.Module):

    def __init__(self, scaling, num_heads, head_dim, model_dim, dropout_rate):
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

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, values, mask):
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

        x = self.attention(queries, keys, values, mask)

        x = x.permute(0, 2, 1, 3).contiguous().view(
            batch_size, n_queries, model_dim)
        
        x = self.final_linear(x)

        x = self.dropout(x)

        return x


class FeedForward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class EncoderLayer(nn.Module):

    def __init__(self, scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(scaling, num_heads, head_dim, model_dim, dropout_rate)
        self.feed_forward = FeedForward(model_dim, hidden_dim, dropout_rate)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, encoder_input):

        residual = encoder_input
        x = self.multi_head_attention(queries=encoder_input, keys=encoder_input, values=encoder_input, mask=None)
        x = self.layer_norm(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm(x + residual)

        return x


class DecoderLayer(nn.Module):

    def __init__(self, scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.masked_multi_head_attention = MultiHeadAttention(scaling, num_heads, head_dim, model_dim, dropout_rate)
        self.multi_head_attention = MultiHeadAttention(scaling, num_heads, head_dim, model_dim, dropout_rate)
        self.feed_forward = FeedForward(model_dim, hidden_dim, dropout_rate)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, encoder_output, decoder_input, mask):

        residual = decoder_input
        x = self.masked_multi_head_attention(queries=decoder_input, keys=decoder_input, values=decoder_input, mask=mask)
        x = self.layer_norm(x + residual)

        residual = x
        x = self.multi_head_attention(x, encoder_output, encoder_output, mask=None)
        x = self.layer_norm(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm(x + residual)

        return x


class EncoderStack(nn.Module):

    def __init__(self, stack_size, scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate):
        super(EncoderStack, self).__init__()

        self.encoder_stack = nn.Sequential()
        for stack in range(stack_size):
            self.encoder_stack.add_module(
                f"encoder_{stack}",
                EncoderLayer(scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate))

    def forward(self, encoder_input):

        encoder_output = self.encoder_stack(encoder_input)

        return encoder_output


class MaskCreator(nn.Module):

    def __init__(self):
        super(MaskCreator, self).__init__()

    def forward(self, sequence_length, device):
        mask = torch.zeros((sequence_length, sequence_length), dtype=torch.float)

        current_col = 1
        for i in range(sequence_length):
            mask[i][current_col:sequence_length] = -float('inf')
            current_col += 1

        return mask.to(device)


class DecoderStack(nn.Module):

    def __init__(self, stack_size, scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate):
        super(DecoderStack, self).__init__()

        self.encoder_stack = nn.ModuleList()
        self.n_stacks = stack_size

        for stack in range(stack_size):
            self.encoder_stack.append(
                DecoderLayer(scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate))

    def forward(self, encoder_output, decoder_value, mask):

        for decoder in self.encoder_stack:
            decoder_value = decoder(encoder_output, decoder_value, mask)

        return decoder_value


class Embeddings(nn.Module):

    def __init__(self, vocab_size, model_dim):
        super(Embeddings, self).__init__()

        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, pre_embedding):

        post_embedding = self.embedding(pre_embedding)

        return post_embedding


class PositionalEncoding(nn.Module):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, seq_len, model_dim, device):

        pos_enc = torch.ones((seq_len, model_dim))
        for seq in range(seq_len):
            for dim in range(model_dim):
                if dim % 2 == 0:
                    pos_enc[seq, dim] = math.sin((2 * seq) / 10000 ** (dim / model_dim))
                else:
                    pos_enc[seq, dim] = math.cos((2 * seq) / 10000 ** (dim / model_dim))

        return pos_enc.to(device)


class DecoderOutputHandler(nn.Module):

    def __init__(self, model_dim, vocab_size):
        super(DecoderOutputHandler, self).__init__()

        self.linear = nn.Linear(model_dim, vocab_size)

    def forward(self, decoder_output):

        transformer_output = self.linear(decoder_output)
        #  transformer_output = F.softmax(transformer_output, dim=-1)

        return transformer_output


class Transformer(nn.Module):
    def __init__(self, stack_size, scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate,
                 encoders_device, decoder_device, vocab_size):

        super(Transformer, self).__init__()

        self.encoder_device = encoders_device
        self.decoder_device = decoder_device
        self.model_dim = model_dim

        self.encoder_embeddings = Embeddings(vocab_size, model_dim).to(encoders_device)
        self.decoder_embeddings = Embeddings(vocab_size, model_dim).to(decoder_device)

        self.positional_encoder = PositionalEncoding()
        self.masker = MaskCreator()

        self.encoder_stack = EncoderStack(stack_size, scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate)\
            .to(encoders_device)
        self.decoder_stack = DecoderStack(stack_size, scaling, num_heads, head_dim, model_dim, hidden_dim, dropout_rate)\
            .to(decoder_device)

        self.decoder_handler = DecoderOutputHandler(model_dim, vocab_size).to(decoder_device)

    def forward(self, encoder_inp, decoder_inp):

        encoder_value = encoder_inp.to(self.encoder_device).long()
        encoder_value = self.encoder_embeddings(encoder_value)
        encoder_value += self.positional_encoder(encoder_value.shape[1], self.model_dim, self.encoder_device)
        encoder_value = self.encoder_stack(encoder_value)

        decoder_value = decoder_inp.to(self.decoder_device).long()
        decoder_value = self.decoder_embeddings(decoder_value)
        decoder_value += self.positional_encoder(decoder_value.shape[1], self.model_dim, self.decoder_device)

        mask = self.masker(decoder_value.shape[1], self.decoder_device)
        encoder_value = encoder_value.to(self.decoder_device)
        decoder_value = self.decoder_stack(encoder_value, decoder_value, mask)

        decoder_value = self.decoder_handler(decoder_value)

        return decoder_value





