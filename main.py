import torch
import tiktoken 
from training import utils
from training import training
import model 
from data import create_dataloader
import json

def main():
    cfg = json.load(open("gpt_124m_config.json", "r"))
    
    text = open("the-verdict.txt", "r").read()
    tokenizer = tiktoken.get_encoding("gpt2")
    train_test_split = int(len(text) * 0.9)
    train_data = text[:train_test_split]
    val_data = text[train_test_split:]
    
    train_loader = create_dataloader(train_data, tokenizer, cfg["context_length"], cfg["stride"], cfg["batch_size"], shuffle=True)
    val_loader = create_dataloader(val_data, tokenizer, cfg["context_length"], cfg["stride"], cfg["batch_size"])

    gpt = model.GPT(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)
    
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=cfg["learning_rate"])

    train_losses, val_losses, track_tokens_seen = training.train_model(gpt, train_loader, val_loader, optimizer, device, cfg["epochs"], cfg["eval_freq"], cfg["eval_iter"], cfg["start_context"], tokenizer)
    
    # Generate epochs_seen for plotting
    epochs_seen = torch.linspace(0, cfg["epochs"], len(train_losses)) # approximate epochs

    torch.save(gpt.state_dict(), "gpt_124m.pt")
    utils.plot_losses(epochs_seen, track_tokens_seen, train_losses, val_losses)

if __name__ == "__main__":
    main()