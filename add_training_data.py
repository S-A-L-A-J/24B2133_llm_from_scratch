import fitz
import torch
import torch.nn as nn
import tiktoken
from important_GPT_blocks import create_dataloader_v1, simple_text_gen, GPT_CONFIG_124M

tokenizer = tiktoken.get_encoding("gpt2")
file_path = "relativity.pdf"
doc = fitz.open(file_path)
text_data = ""
for page in doc:
    text_data += page.get_text()
total_chars = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

train_ratio = 0.90
split_idx = int(train_ratio * total_chars)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(train_data,
                                    batch_size=2,
                                    max_length=GPT_CONFIG_124M['context_length'],
                                    stride=GPT_CONFIG_124M['context_length'],
                                    drop_last=True,
                                    shuffle=True,
                                    num_workers=0
                                    )

val_loader = create_dataloader_v1(val_data,
                                    batch_size=2
                                    ,
                                    max_length=GPT_CONFIG_124M['context_length'],
                                    stride=GPT_CONFIG_124M['context_length'],
                                    drop_last=False,
                                    shuffle=False,
                                    num_workers=0
                                    )

def loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def loss_batch_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else : 
            break
    return total_loss/num_batches


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_seen_tokens = [], [], []
    tokens_count, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_count += input_batch.numel()
            global_step += 1
            
        if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_seen_tokens.append(tokens_count)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_seen_tokens

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = loss_batch_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = loss_batch_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = simple_text_gen(
            model=model, idx=encoded,
            max_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) 
    model.train()