import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TextDataset
from torch.distributions.categorical import Categorical

def scaled_softmax_attention(query, key, value, mask=None):
    """
    Args:
        query: torch.Tensor (..., L, D)
        key: torch.Tensor (..., L, D)
        value: torch.Tensor (..., L, D)
    Returns:
        res: torch.Tensor (..., L, D), output of the attention layer (\softmax(Q K^T / d) V
        attention: torch.Tensor (..., L, L), attention weights (\softmax(Q K^T / d))

    L is the length of sequence, D is the embedding dimension
    """
    scaled_kq = (query @ key.transpose(-1, -2)) / (key.shape[-1]) ** (1/2)
    if mask is not None:
        scaled_kq = (1 - mask) * scaled_kq
    attention =  F.softmax(scaled_kq, dim=-1)
    res = attention @ value

    return res, attention


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    # original implementation uses this initialization
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, return_attention=False, mask=None):
        """
        Args:
            x: torch.Tensor (B, L, D)
            return_attention: If specified, returns attention along with outputs
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)

        B is batch size, L is the length of sequence, D is the embedding dimension
        """
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        batch_size, len, _ = x.shape
        query = query.reshape(batch_size, len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, len, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        res, attention = scaled_softmax_attention(query, key, value, mask=mask)
        res = res.transpose(1, 2).reshape(batch_size, len, self.embed_dim)
        outputs = self.o_proj(res)

        if return_attention:
            return outputs, attention
        else:
            return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        pe = torch.zeros(1, max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).reshape(-1,1)
        div = 10000.0 ** (torch.arange(0, embed_dim, 2, dtype=torch.float) / embed_dim)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

        pe[:, :, 0::2] = torch.sin((pos / div))
        pe[:, :, 1::2] = torch.cos((pos / div))


    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, feedforward_dim, activation=nn.ReLU, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.activation = activation()

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, return_attention=False, mask=None):
        """
        Args:
            x: torch.Tensor (B, L, D)
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)
        """

        outputs, attention = self.mha(x, return_attention=True, mask=mask)

        outputs = self.layer_norm(x + outputs)
        outputs = self.activation(self.linear1(outputs))
        outputs = self.linear2(outputs)

        outputs = self.layer_norm(x + outputs)

        if return_attention:
            return outputs, attention
        else:
            return outputs
        


def get_non_pad_mask(seq, pad_id):
    assert seq.dim() == 2
    return seq.ne(pad_id).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, pad_id):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_lookahead_mask(shape):
    # Mask out future entries by marking them with a 1.0
    return torch.triu(torch.ones((shape, shape)), diagonal=1)

class TransformerDecoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dataset: TextDataset,
        feedforward_dim: int = 256,
        num_layers: int = 1,
        activation = nn.ReLU,
        dropout: float = 0.0
    ):
        super().__init__()
        # define layers
        self.dataset = dataset 
        self.pad_id = self.dataset.pad_id
        self.input_embedding = nn.Embedding(
            num_embeddings=dataset.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=self.pad_id
        )
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=dataset.max_length)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim=embed_dim, num_heads=num_heads, feedforward_dim=feedforward_dim, activation=activation, dropout=dropout)])

        self.linear = nn.Linear(embed_dim, self.vocab_size)

    def to(self, device, **kwargs):
      self.device = device
      return super().to(device, **kwargs)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x, pad_id=self.pad_id)
        lookahead_mask = get_lookahead_mask(shape=x.shape[1])
        lookahead_mask = torch.max(slf_attn_mask, lookahead_mask)

        for dec_block in self.decoder_blocks:
            x = dec_block(x, mask=lookahead_mask)
        logits = self.linear(x)
        return logits
    
    
    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        # encode prefix
        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        # generate hidden for prefix
        tokens = self.input_embedding(tokens)
        tokens = self.positional_encoding(tokens)

        for dec_block in self.decoder_blocks:
            tokens = dec_block(tokens)
        logits = self.linear(tokens) / temp

        # sample new token from logits
        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        # 2 stopping conditions: reaching max len or getting <eos> token
        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            # process newly obtained token
            tokens = self.input_embedding(tokens)
            tokens = self.positional_encoding(tokens)

            for dec_block in self.decoder_blocks:
                tokens = dec_block(tokens)
            logits = self.linear(tokens) / temp
            # sample the next token from logits
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        # decode result to a string
        return self.dataset.ids2text(tokens.squeeze())
