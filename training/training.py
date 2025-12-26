import torch
from model import GPT
from data import create_dataloader
from utils import calculate_loss_batch, evaluate_model, generate_and_print_sample
import json

def get_data():
    with open("the-verdict.txt", "r") as f:
        text = f.read()
    data = create_dataloader(text)
    return data


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    
    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            model.train()
            optimizer.zero_grad()
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch}, Global Step {global_step}, Train Loss {train_loss:.4f}, Validation Loss {val_loss:.4f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen




    

