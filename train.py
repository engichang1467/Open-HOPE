import argparse
import os
import yaml
import torch
import lightning as L
# from src.models.hope import HOPE
# from src.utils.trainer import MultiFrequencyTrainer
from src.utils.data_loader import get_data_loader
from src.utils.lightning_trainer import HOPELightningModule


# Set the environment variable to prevent tokenizer parallelism issues with DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description="Train HOPE model with PyTorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="wikitext-2", help="Dataset name")
    parser.add_argument("--limit", type=int, default=1024, help="Limit dataset size for debugging")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # The Lightning Trainer will handle device placement.
    # We can specify accelerator='gpu' and devices=1 to use a single GPU.
    # 'auto' is also a great option.
    
    # Initialize Lightning Module
    print("Initializing HOPE LightningModule...")
    model_module = HOPELightningModule(config)
    
    # Data Loader
    print(f"Loading dataset {args.dataset}...")
    dataloader = get_data_loader(
        args.dataset, 
        config['training']['batch_size'], 
        config['model']['max_seq_len'],
        limit=args.limit
    )
    
    # Trainer
    print("Setting up Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator="auto",
        devices="auto",
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model_module, dataloader)
    
    # Save model
    # The Lightning Trainer automatically saves checkpoints.
    # To save the final model manually:
    torch.save(model_module.model.state_dict(), "hope_model_lightning.pth")
    print("Training complete. Model saved to hope_model_lightning.pth")

if __name__ == "__main__":
    main()
