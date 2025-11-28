import argparse
import yaml
import torch
import math
from tqdm import tqdm
from src.models.hope import HOPE
from src.utils.data_loader import get_data_loader

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            logits = model(batch)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def main():
    parser = argparse.ArgumentParser(description="Evaluate HOPE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, default="hope_model.pth", help="Path to saved model")
    parser.add_argument("--dataset", type=str, default="wikitext-2", help="Dataset name")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HOPE(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    dataloader = get_data_loader(
        args.dataset, 
        config['training']['batch_size'], 
        config['model']['max_seq_len'],
        split='test' # or validation
    )
    
    print("Starting evaluation...")
    loss, ppl = evaluate(model, dataloader, device)
    print(f"Evaluation Results - Loss: {loss:.4f}, Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
