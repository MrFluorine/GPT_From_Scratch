import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32    # How many independent sequence will be processed in parallel?
block_size = 8     # What is the maximum context length for predictions?
max_iters = 5000   
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
#------------------

torch.manual_seed(1377)
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt",'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# creating a mapping from characters to integers and vice versa
s_to_i = {ch:i for i, ch in enumerate(chars)}
i_to_s = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [s_to_i[ch] for ch in s]
decode = lambda l: ''.join([i_to_s[i] for i in l])

# train and validation split
data = torch.tensor(encode(text), dtype= torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading 
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y =  x.to(device), y.to(device)
    return x, y
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    """ one head of the self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)           #(B, T, head_size)
        q = self.query(x)         #(B, T, head_size)
        # compute the attention scores ("affinities")
        wei = q @ k.transpose(-2, -1)*C**-0.5      #(B, T, head_size) @ (B, head_size, T)----->(B, head_size, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        #perform the weighted aggeration of the values
        v = self.value(x)         #(B, T, head_size)
        out = wei @ v             #(B, head_size, T) @ (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non linearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential( nn.Linear(n_embed, 4*n_embed),
                                 nn.ReLU(),
                                 nn.Linear(4*n_embed, n_embed),
                    )
    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimenssion, n_head: number of attention heads we would like
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
    
    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
    
# super simple bigram model
class BigramLanguageModel(nn.Module):


    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed )
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
    

    def forward(self,idx, targets=None):
        B, T = idx.shape
        #idx and targets are both (B,T) tensor of integers 
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  #(T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)          #(B, T, C)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_token):
        #idx is a (B,T) array of indices in current context
        for _ in range(max_new_token):
            # crop the context

            idx_const = idx[:, -block_size:]
            #get the prediction
            logits, loss = self(idx_const) #(B, T, C)
            #focus only in the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            #apply softmax to get probablities
            probs = F.softmax(logits, dim =-1) #(B,C)
            #sample for distributyion
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sample imput to the runing sequence
            idx = torch.cat((idx, idx_next), dim =1) #(B, T+1)
        return idx
model = BigramLanguageModel()
m = model.to(device)

#Create a Pytorch optimiser
optimiser = torch.optim.Adam(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f},val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# generate from model
context = torch.zeros((2,3), dtype=torch.long, device=device)

print(decode(m.generate(context, max_new_token=500)[0].tolist()))