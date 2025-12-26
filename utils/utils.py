import torch 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    model.eval()

    for i in range(max_new_tokens):
        idx = idx[:, -context_size:]
        logits = model(idx)
        logits = logits[:, -1, :] # get logits of last character in each batch 
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')),
                logits
            )
        if temperature > 0:
            logits = logits / temperature # greater temperature evens the distribution more resulting in more diverse sampling (probabilities are similar)
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    
    model.train()
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context = model.pos_emb.weight.shape[0]
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_ids = generate(model, encoded, 50, context)
    decoded = tokenizer.decode(generated_ids.squeeze(0).tolist())
    print(decoded.replace("\n", " "))


def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calculate_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calculate_loss_loader(val_loader, model, device, num_batches=eval_iter)
    return train_loss, val_loss


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Train Loss")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Tokens Seen")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, color="tab:blue", label="Tokens Seen")
    ax2.set_xlabel("Tokens Seen")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()
