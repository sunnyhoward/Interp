import torch as t
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=1, dim_feedforward=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(10, d_model)  # max sequence length ~10
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.nhead = nhead

    def forward(self, x):
        seq_len = x.size(1)
        positions = t.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.out(x)
        return logits


class AttentionBlock(nn.Module):
    """Single attention layer with residual connection"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Attention with residual connection
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        x = x + attn_out  # Residual stream
        x = self.ln(x)
        return x, attn_weights


class StackedAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(10, d_model)
        
        # Stack of attention layers
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, nhead) 
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = t.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        
        # Pass through each attention layer in the residual stream
        for layer in self.layers:
            x, _ = layer(x)
        logits = self.out(x)
        return logits
    
    def greedy(self,logits,dataset):
        idxs =  t.argmax(logits, dim=-1)
        return dataset.idx_to_str(idxs[0])
    
    


class AdditionDataset(Dataset):
    def __init__(self, n_samples=20000, min_val = 0, max_val=99, no_test_examples=200, no_digits_to_predict=None): #if none then predict all.
        self.samples = []
        self.no_digits_to_predict = no_digits_to_predict

        assert max_val <= 999  # Support up to 3 digits now
        assert no_test_examples <= (max_val+1)**2

        # choose held-out test pairs
        self.test_examples = set()
        while len(self.test_examples) < no_test_examples:
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
            self.test_examples.add((a, b))
        self.test_examples = list(self.test_examples)

        # build training pairs excluding test pairs
        for _ in range(n_samples):
            while True:
                a = random.randint(min_val, max_val)
                b = random.randint(min_val, max_val)
                if (a, b) not in self.test_examples:
                    break
            inp, out = self.encode_pair(a, b)
            self.samples.append((inp, out))

        self.test_samples = []
        for a, b in self.test_examples:
            inp, out = self.encode_pair(a, b)
            self.test_samples.append((inp, out))

        # token set: digits + symbols + padding
        self.vocab = {str(i): i for i in range(10)}
        self.vocab['+'] = 10
        self.vocab['='] = 11
        self.vocab['<PAD>'] = 12
        
        self.rev_vocab = {i: tok for tok, i in self.vocab.items()}
        
        # Calculate max lengths for padding
        self.max_input_len = max(len(inp) for inp, _ in self.samples)
        self.max_output_len = max(len(out) for _, out in self.samples)

        self.train = True    


    def encode_pair(self, a, b):
        s = a + b
        
        # Convert to strings without leading zeros
        a_str = str(a)
        b_str = str(b)

        s_str = str(s)
        if self.no_digits_to_predict:
            s_str = s_str[:self.no_digits_to_predict]
        
        # Build sequences
        inp = list(a_str) + ['+'] + list(b_str) + ['=']
        out = list(s_str)
        
        return inp, out
    
    def pad_pair(self, inp, out):
        inp += ['<PAD>'] * (self.max_input_len - len(inp))
        out += ['<PAD>'] * (self.max_output_len - len(out))

        # Convert to indices
        inp_indices = [self.vocab[x] for x in inp]
        out_indices = [self.vocab[x] for x in out]
        
        # # Pad sequences
        # inp_indices += [self.vocab['<PAD>']] * (self.max_input_len - len(inp_indices))
        # out_indices += [self.vocab['<PAD>']] * (self.max_output_len - len(out_indices))
        
        inp_tensor = t.tensor(inp_indices, dtype=t.long)
        out_tensor = t.tensor(out_indices, dtype=t.long)
        
        return inp_tensor, out_tensor

    def __len__(self):
        return len(self.samples) if self.train else len(self.test_samples)

    def __getitem__(self, idx):
        inp, out = self.samples[idx] if self.train else self.test_samples[idx]
        inp_tensor, out_tensor = self.pad_pair(inp, out)
        
        return inp_tensor, out_tensor

    def idx_to_str(self, idxs:t.tensor):
        if len(idxs.shape) == 1:
            return [self.rev_vocab[i.item()] for i in idxs]# if i.item() != self.vocab['<PAD>']]
        elif len(idxs.shape) == 2:
            return [[self.rev_vocab[i.item()] for i in idx] for idx in idxs]
    
    def get_example(self, a:int, b:int):
        inp, out = self.encode_pair(a, b)
        inp_tensor, out_tensor = self.pad_pair(inp, out)

        return inp_tensor[None], out_tensor[None]