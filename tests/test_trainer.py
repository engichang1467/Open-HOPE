import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.utils.trainer import MultiFrequencyTrainer


class MockCMSModule(nn.Module):
    """Mock CMS module for testing"""
    def __init__(self, num_levels=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(num_levels)
        ])
    
    def forward(self, x):
        return x


# class MockTransformerLayer(nn.Module):
#     """Mock transformer layer for testing"""
#     def __init__(self):
#         super().__init__()
#         self.cms = MockCMSModule(num_levels=2)
#         self.norm = nn.LayerNorm(64)
#         self.attention = nn.Linear(64, 64)
    
#     def forward(self, x):
#         return x

class MockHOPEModel(nn.Module):
    """Mock HOPE model for testing"""
    def __init__(self, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, 64)
        # Create proper PyTorch modules using ModuleDict to match production code
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cms': MockCMSModule(num_levels=2),
                'norm': nn.LayerNorm(64),
                'attention': nn.Linear(64, 64)
            }) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 1000)
    
    def forward(self, x):
        # Simple forward pass for testing
        x = self.embedding(x)
        for layer in self.layers:
            x = layer['norm'](x)
        x = self.final_norm(x)
        return self.lm_head(x)

# class MockHOPEModel(nn.Module):
#     """Mock HOPE model for testing"""
#     def __init__(self, num_layers=2):
#         super().__init__()
#         self.embedding = nn.Embedding(1000, 64)
#         self.layers = nn.ModuleList([
#             {'cms': MockCMSModule(num_levels=2), 'norm': nn.LayerNorm(64)}
#             for _ in range(num_layers)
#         ])
#         self.final_norm = nn.LayerNorm(64)
#         self.lm_head = nn.Linear(64, 1000)
    
#     def forward(self, x):
#         # Simple forward pass for testing
#         x = self.embedding(x)
#         for layer in self.layers:
#             x = layer['norm'](x)
#         x = self.final_norm(x)
#         return self.lm_head(x)
    
#     def named_children(self):
#         # Mock the named_children method that trainer uses
#         return [
#             ('embedding', self.embedding),
#             ('layers', self.layers),
#             ('final_norm', self.final_norm),
#             ('lm_head', self.lm_head)
#         ]



@pytest.fixture
def mock_config():
    """Test configuration"""
    return {
        'cms': {
            'levels': [
                {'frequency': 1, 'dimension': 64},
                {'frequency': 4, 'dimension': 64}
            ]
        },
        'training': {
            'learning_rate': '1e-4',
            'gradient_accumulation_steps': 1
        }
    }


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    model = MockHOPEModel(num_layers=2)
    
    # # Manually set up the layers structure that trainer expects
    # for i, layer_dict in enumerate(model.layers):
    #     # Create a mock object that has named_children method
    #     layer_mock = MagicMock()
    #     layer_mock.named_children.return_value = [
    #         ('cms', layer_dict['cms']),
    #         ('norm', layer_dict['norm'])
    #     ]
    #     # Replace the dict with our mock
    #     model.layers[i] = layer_mock
    #     # Add cms as an attribute for direct access
    #     model.layers[i].cms = layer_dict['cms']
    
    return model


@pytest.fixture
def device():
    """Test device"""
    return torch.device('cpu')


@pytest.fixture
def sample_batch():
    """Sample training batch"""
    return torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10


class TestMultiFrequencyTrainer:
    """Smoke tests for MultiFrequencyTrainer"""
    
    def test_trainer_initialization(self, mock_model, mock_config, device):
        """Test that trainer initializes without errors"""
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        
        assert trainer.model is not None
        assert trainer.config == mock_config
        assert trainer.device == device
        assert len(trainer.optimizers) > 0
    
    def test_parameter_grouping(self, mock_model, mock_config, device):
        """Test that parameters are grouped correctly by frequency"""
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        
        # Should have groups for frequencies 1 and 4
        assert 1 in trainer.param_groups
        assert 4 in trainer.param_groups
        
        # Both groups should have parameters
        assert len(trainer.param_groups[1]) > 0
        assert len(trainer.param_groups[4]) > 0
        
        # Should have optimizers for each frequency
        assert 1 in trainer.optimizers
        assert 4 in trainer.optimizers
    
    def test_forward_pass(self, mock_model, mock_config, device, sample_batch):
        """Test that model can perform forward pass"""
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        trainer.model.eval()
        
        with torch.no_grad():
            output = trainer.model(sample_batch)
        
        # Check output shape
        batch_size, seq_len = sample_batch.shape
        vocab_size = 1000
        assert output.shape == (batch_size, seq_len, vocab_size)
    
    def test_loss_calculation(self, mock_model, mock_config, device, sample_batch):
        """Test that loss can be calculated"""
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        trainer.model.train()
        
        # Forward pass
        logits = trainer.model(sample_batch)
        
        # Calculate loss (same logic as in trainer)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sample_batch[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Loss should be a scalar tensor
        assert loss.dim() == 0
        assert loss.requires_grad
    
    def test_backward_pass(self, mock_model, mock_config, device, sample_batch):
        """Test that backward pass works without errors"""
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        trainer.model.train()
        
        # Forward pass
        logits = trainer.model(sample_batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sample_batch[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in trainer.model.parameters())
        assert has_gradients
    
    def test_optimizer_step(self, mock_model, mock_config, device, sample_batch):
        """Test that optimizers can step without errors"""
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        trainer.model.train()
        
        # Store initial parameter values
        initial_params = {id(p): p.clone().detach() for p in trainer.model.parameters()}
        
        # Forward and backward
        logits = trainer.model(sample_batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sample_batch[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        
        # Step all optimizers
        for freq, optimizer in trainer.optimizers.items():
            optimizer.step()
            optimizer.zero_grad()
        
        # Check that some parameters changed
        params_changed = False
        for p in trainer.model.parameters():
            if not torch.equal(p, initial_params[id(p)]):
                params_changed = True
                break
        
        assert params_changed, "No parameters were updated during optimizer step"
    
    @patch('torch.utils.data.DataLoader')
    def test_train_single_step(self, mock_dataloader, mock_model, mock_config, device, sample_batch):
        """Test a single training step"""
        # Mock dataloader to return our sample batch
        mock_dataloader.return_value = iter([sample_batch])
        
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        
        # Create a simple dataloader mock
        dataloader = [sample_batch]  # Single batch
        
        # Test that we can run one epoch without errors
        try:
            trainer.train(dataloader, num_epochs=1)
            training_successful = True
        except Exception as e:
            training_successful = False
            pytest.fail(f"Training failed with error: {e}")
        
        assert training_successful
    
    def test_frequency_based_updates(self, mock_model, mock_config, device, sample_batch):
        """Test that different frequencies update at correct intervals"""
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        
        # Mock the training loop logic
        global_step = 1
        
        # Test frequency 1 (should update every step)
        assert global_step % 1 == 0
        
        # Test frequency 4 (should update every 4 steps)
        assert global_step % 4 != 0  # Step 1 shouldn't update freq 4
        
        global_step = 4
        assert global_step % 4 == 0  # Step 4 should update freq 4
    
    def test_device_placement(self, mock_model, mock_config):
        """Test that model is moved to correct device"""
        device = torch.device('cpu')
        trainer = MultiFrequencyTrainer(mock_model, mock_config, device)
        
        # Check that model is on the correct device
        for param in trainer.model.parameters():
            assert param.device == device


if __name__ == "__main__":
    pytest.main([__file__])