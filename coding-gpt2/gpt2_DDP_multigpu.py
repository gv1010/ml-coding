import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
import tiktoken
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 512, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadCasualAttention(
            d_in= cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # This block does nothing and just returns its input.
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        # This layer does nothing and just returns its input.

        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim=-1,keepdim = True, unbiased = False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadCasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


# Assuming you have your GPT-2 model defined already
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # print(in_idx.shape)
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits

# Custom Dataset (assuming you have this)
class CustomDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = CustomDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    # dataloader = DataLoader(
    #     dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataset


def load_train_val_dataset():
    from datasets import load_dataset

    train_dataset = load_dataset('flytech/python-codes-25k', split='train')
    # train_dataset = load_dataset('flytech/python-codes-25k', split='test')

    # One can map the dataset in any way, for the sake of example:
    dataset = train_dataset.map(lambda example: {'text': example['instruction'] + ' ' + example['input'] + ' ' + example['output']})['text']
    # Remember that you don't need to map if the dataset has a "text" field already:)
    train_ratio = 0.90
    # text_data = "<|endoftext|>".join(dataset[:50])
    dataset = dataset[:]
    split_idx = int(train_ratio * len(dataset))

    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    train_data = "<|endoftext|>".join(train_data[:])
    val_data = "<|endoftext|>".join(val_data[:])
    # print("split_idx: ", split_idx)

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=8,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=8,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    print("length of dataset", len(train_data), len(val_data))
    print("length of loader", len(train_loader), len(val_loader))
    return train_loader, val_loader

def text_to_token_ids(text, tokenizer):
    # encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    try:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss
    except Exception as exp:
        print(exp)
        return 0

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def setup_ddp(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def plot_losses(train_losses, val_losses, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss (per batch)')
    plt.plot([i * (len(train_losses) // epochs) for i in range(epochs + 1)], [val_losses[0]] + val_losses, label='Validation Loss (per epoch)', marker='o')
    plt.xlabel('Batch / Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def train(rank, world_size, batch_size=8, epochs=10):
    """Training function with DDP"""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create model and move to device
    model = GPTModel(GPT_CONFIG_124M).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    
    # Prepare data
    train_data, val_data = load_train_val_dataset()
    # dataset = CustomDataset(train_data)
    # Use DistributedSampler for DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4
    )

    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_data,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=False
    # )
    # val_loader = DataLoader(
    #     val_data,
    #     batch_size=batch_size,
    #     sampler=val_sampler,
    #     num_workers=4,
    #     drop_last=False
    # )
    print("val_data: ", len(val_data))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    
    if rank == 0:
        plt.ion()
        
    # Training loop
    model.train()
    training_loss = []
    validation_loss = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Ensure shuffling works properly across epochs
        
        total_loss = 0
        batch_idx = 0
        for input_batch, target_batch in (train_dataloader):
            optimizer.zero_grad()
            batch_idx += 1

            loss = calc_loss_batch(input_batch, target_batch, model, device)  # Assuming your model returns a loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        if rank == 0:
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch} completed. Average Train Loss: {avg_loss:.4f}')
            training_loss.append(avg_loss)

            # total_val_loss = 0
            # val_idx = 0
            # for input_batch, target_batch in (val_loader):
            #     model.eval()
            #     val_loss = calc_loss_batch(input_batch, target_batch, model, device)
            #     val_idx += 1
            #     total_val_loss += (val_loss.item())

            # if val_idx:
            #     val_loss = total_val_loss/(val_idx)
            #     validation_loss.append(val_loss)
            #     print(f'Epoch {epoch} completed. Validation Loss: {val_loss:.4f}')

    
    # Save model (only from rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), 'gpt2_ddp_model_2.pth')
        torch.save({"train_loss": training_loss, "validation_loss": validation_loss}, "loss_gpt2_ddp_model_2.pth")
        # plot_losses(training_loss, validation_loss, epoch + 1)
        # plt.ioff() # Turn off interactive mode
        # plt.show()
    
    # Cleanup
    cleanup()

def main():
    # Model configuration
    GPT_CONFIG_124M = {
                        "vocab_size": 50257,   # Vocabulary size
                        "context_length": 512, # Shortened context length (orig: 1024)
                        "emb_dim": 768,        # Embedding dimension
                        "n_heads": 12,         # Number of attention heads
                        "n_layers": 12,        # Number of layers
                        "drop_rate": 0.1,      # Dropout rate
                        "qkv_bias": False      # Query-key-value bias
                    }

    world_size = torch.cuda.device_count()
    
    # Launch DDP training
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Set multiprocessing start method
    torch.multiprocessing.set_start_method('spawn')
    main()
