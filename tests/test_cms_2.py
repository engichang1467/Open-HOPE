import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.utils.trainer import MultiFrequencyTrainer

class MockCMS(nn.Module):
    def __init__(self, levels_config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in levels_config:
            self.layers.append(nn.Linear(10, 10))

class MockLayer(nn.Module):
    def __init__(self, levels_config):
        super().__init__()
        self.cms = MockCMS(levels_config)
        self.other = nn.Linear(10, 10)
        
    def __getitem__(self, key):
        if key == 'cms':
            return self.cms
        return getattr(self, key)

class MockModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(config['cms']['levels'])])
        self.embedding = nn.Embedding(10, 10)
        self.final_norm = nn.LayerNorm(10)
        self.lm_head = nn.Linear(10, 10)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            # Pass through CMS layers sequentially for mock
            for cms_layer in layer.cms.layers:
                x = cms_layer(x)
            x = layer.other(x)
            
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x

def test_cms_update_logic():
    config = {
        'cms': {
            'levels': [
                {'name': 'fast', 'chunk_size': 2},
                {'name': 'slow', 'chunk_size': 4},
                {'name': 'static', 'chunk_size': 0}
            ]
        },
        'training': {
            'learning_rate': 0.1,
            'gradient_accumulation_steps': 1
        }
    }
    
    model = MockModel(config)
    trainer = MultiFrequencyTrainer(model, config, device='cpu')
    
    # Identify parameters
    fast_params = list(model.layers[0].cms.layers[0].parameters())
    slow_params = list(model.layers[0].cms.layers[1].parameters())
    static_params = list(model.layers[0].cms.layers[2].parameters())
    other_params = list(model.layers[0].other.parameters())
    
    # Helper to get param values
    def get_params(params):
        return [p.clone() for p in params]
    
    # Initial state
    fast_0 = get_params(fast_params)
    slow_0 = get_params(slow_params)
    static_0 = get_params(static_params)
    other_0 = get_params(other_params)
    
    # Dummy dataloader
    # Batch size 2, Sequence length 10
    # Values between 0 and 10 (vocab size)
    dataloader = [torch.randint(0, 10, (2, 10)) for _ in range(10)] # 10 batches
    
    # We need to manually step the trainer logic or mock the loop.
    # Let's use the trainer.train but with a mocked dataloader and check state after steps.
    # However, trainer.train runs the whole loop. 
    # Let's just manually invoke the update logic or run for 1 epoch with 5 steps.
    
    # Step 1
    # Global step 1.
    # Fast (2): No update (1 % 2 != 0)
    # Slow (4): No update
    # Static (0): No update
    # Other (1): Update (1 % 1 == 0)
    
    # We can't easily hook into the loop without modifying it or using hooks.
    # But we can check the weights after the loop runs for N steps.
    
    # Let's run for 2 steps (batch_size=1, accum=1)
    # Step 1: Other updates. Fast/Slow/Static don't.
    # Step 2: Other updates. Fast updates. Slow/Static don't.
    
    # To verify "don't update", we need to ensure gradients are non-zero but weights didn't change?
    # Or just weights didn't change.
    
    # Let's run the training loop for 2 steps.
    trainer.train(dataloader[:2], num_epochs=1)
    
    fast_2 = get_params(fast_params)
    slow_2 = get_params(slow_params)
    static_2 = get_params(static_params)
    other_2 = get_params(other_params)
    
    # Check Step 2
    # Fast should have changed (updated at step 2)
    assert not torch.allclose(fast_0[0], fast_2[0]), "Fast params should have updated at step 2"
    
    # Slow should NOT have changed (updates at step 4)
    assert torch.allclose(slow_0[0], slow_2[0]), "Slow params should NOT have updated at step 2"
    
    # Static should NOT have changed
    assert torch.allclose(static_0[0], static_2[0]), "Static params should NOT have updated"
    
    # Other should have changed
    assert not torch.allclose(other_0[0], other_2[0]), "Other params should have updated"
    
    print("Test passed: Step 2 behavior correct.")
    
    # Run 2 more steps (Total 4)
    trainer.train(dataloader[2:4], num_epochs=1)
    
    fast_4 = get_params(fast_params)
    slow_4 = get_params(slow_params)
    
    # Fast should have changed again (updated at step 4)
    assert not torch.allclose(fast_2[0], fast_4[0]), "Fast params should have updated at step 4"
    
    # Slow should have changed (updated at step 4)
    assert not torch.allclose(slow_2[0], slow_4[0]), "Slow params should have updated at step 4"
    
    print("Test passed: Step 4 behavior correct.")

if __name__ == "__main__":
    test_cms_update_logic()
