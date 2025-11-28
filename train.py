import argparse
import yaml
import torch
from src.models.hope import HOPE
from src.utils.data_loader import get_data_loader
from src.utils.trainer import MultiFrequencyTrainer

def main():
    parser = argparse.ArgumentParser(description="Train HOPE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="wikitext-2", help="Dataset name")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    print("Initializing HOPE model...")
    model = HOPE(config)
    
    # Data Loader
    print(f"Loading dataset {args.dataset}...")
    dataloader = get_data_loader(
        args.dataset, 
        config['training']['batch_size'], 
        config['model']['max_seq_len']
    )
    
    # Trainer
    print("Setting up trainer...")
    trainer = MultiFrequencyTrainer(model, config, device)
    
    # Train
    print("Starting training...")
    trainer.train(dataloader, config['training']['num_epochs'])
    
    # Save model
    torch.save(model.state_dict(), "hope_model.pth")
    print("Training complete. Model saved to hope_model.pth")

if __name__ == "__main__":
    main()
