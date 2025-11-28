import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.models.hope import HOPE
from src.optimizers.internal_opt import InternalOptimizer


@pytest.fixture
def config():
    """Basic configuration for testing."""
    return {
        'model': {
            'dim': 128,
            'depth': 2,
            'vocab_size': 1000,
            'num_heads': 4
        },
        'cms': {
            'levels': [
                {'frequency': 1},  # Update every step
                {'frequency': 4},  # Update every 4 steps
                {'frequency': 16}, # Update every 16 steps
            ]
        }
    }


@pytest.fixture
def sample_input():
    """Sample input tensor for testing."""
    batch_size = 2
    seq_len = 8
    vocab_size = 1000
    return torch.randint(0, vocab_size, (batch_size, seq_len))


class TestHOPE:
    """Smoke tests for HOPE architecture."""
    
    def test_hope_initialization(self, config):
        """Test that HOPE model can be initialized without errors."""
        model = HOPE(config)
        assert isinstance(model, HOPE)
        assert model.dim == config['model']['dim']
        assert model.depth == config['model']['depth']
        assert model.vocab_size == config['model']['vocab_size']
    
    def test_hope_forward_pass(self, config, sample_input):
        """Test that HOPE can perform a forward pass."""
        model = HOPE(config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
        
        batch_size, seq_len = sample_input.shape
        expected_shape = (batch_size, seq_len, config['model']['vocab_size'])
        assert output.shape == expected_shape
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    def test_hope_gradient_flow(self, config, sample_input):
        """Test that gradients flow through the model properly."""
        model = HOPE(config)
        model.train()
        
        # Simple loss computation
        output = model(sample_input)
        loss = F.cross_entropy(
            output.view(-1, config['model']['vocab_size']),
            sample_input.view(-1)
        )
        
        # Check that loss is finite
        assert torch.isfinite(loss), "Loss is NaN or Inf"
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"
    
    def test_titans_component(self, config):
        """Test the Self-Modifying Titans component."""
        from src.models.titan import SelfModifyingTitans
        
        dim = config['model']['dim']
        head_dim = dim // config['model']['num_heads']
        titans = SelfModifyingTitans(dim, head_dim)
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, dim)
        
        output, state = titans(x)
        
        assert output.shape == x.shape
        assert state.shape == (batch_size, head_dim, head_dim)
        assert torch.isfinite(output).all()
        assert torch.isfinite(state).all()
    
    def test_cms_component(self, config):
        """Test the CMS component."""
        from src.memory.cms import CMS
        
        dim = config['model']['dim']
        cms = CMS(dim, config['cms']['levels'])
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, dim)
        
        output = cms(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_cms_frequency_updates(self, config):
        """Test CMS frequency-based parameter updates."""
        from src.memory.cms import CMS
        
        dim = config['model']['dim']
        cms = CMS(dim, config['cms']['levels'])
        
        # Test different steps
        step_1_params = cms.get_parameters_by_frequency(1)
        step_4_params = cms.get_parameters_by_frequency(4)
        step_16_params = cms.get_parameters_by_frequency(16)
        
        # Step 1 should update level 0 (freq=1)
        assert len(step_1_params) > 0
        
        # Step 4 should update levels 0 and 1 (freq=1 and freq=4)
        assert len(step_4_params) > len(step_1_params)
        
        # Step 16 should update all levels
        assert len(step_16_params) > len(step_4_params)
    
    def test_internal_optimizer(self):
        """Test the internal optimizer component."""
        head_dim = 32
        optimizer = InternalOptimizer(head_dim)
        
        batch_size = 2
        W_t = torch.randn(batch_size, head_dim, head_dim)
        x_t = torch.randn(batch_size, head_dim, 1)
        grad_L = torch.randn(batch_size, head_dim, 1)
        
        W_t_plus_1 = optimizer(W_t, x_t, grad_L)
        
        assert W_t_plus_1.shape == W_t.shape
        assert torch.isfinite(W_t_plus_1).all()
        
        # Check that W actually changed
        assert not torch.equal(W_t, W_t_plus_1)
    
    def test_hope_cms_integration(self, config, sample_input):
        """Test integration between HOPE and CMS parameter updates."""
        model = HOPE(config)
        
        # Test parameter collection at different steps
        params_step_1 = model.get_cms_parameters_by_frequency(1)
        params_step_4 = model.get_cms_parameters_by_frequency(4)
        params_step_16 = model.get_cms_parameters_by_frequency(16)
        
        assert len(params_step_1) > 0
        assert len(params_step_4) >= len(params_step_1)
        assert len(params_step_16) >= len(params_step_4)
    
    def test_different_sequence_lengths(self, config):
        """Test HOPE with different sequence lengths."""
        model = HOPE(config)
        model.eval()
        
        batch_size = 2
        vocab_size = config['model']['vocab_size']
        
        for seq_len in [1, 4, 8, 16]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
            
            expected_shape = (batch_size, seq_len, vocab_size)
            assert output.shape == expected_shape
            assert torch.isfinite(output).all()
    
    def test_different_batch_sizes(self, config):
        """Test HOPE with different batch sizes."""
        model = HOPE(config)
        model.eval()
        
        seq_len = 8
        vocab_size = config['model']['vocab_size']
        
        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
            
            expected_shape = (batch_size, seq_len, vocab_size)
            assert output.shape == expected_shape
            assert torch.isfinite(output).all()


class TestHOPEEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_cms_levels(self, config):
        """Test CMS with empty levels configuration."""
        from src.memory.cms import CMS
        
        config_empty = config.copy()
        config_empty['cms']['levels'] = []
        
        cms = CMS(config['model']['dim'], config_empty['cms']['levels'])
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config['model']['dim'])
        
        # Should work but just return input
        output = cms(x)
        assert torch.equal(output, x)
    
    def test_single_token_sequence(self, config):
        """Test HOPE with single token sequences."""
        model = HOPE(config)
        model.eval()
        
        batch_size = 2
        vocab_size = config['model']['vocab_size']
        input_ids = torch.randint(0, vocab_size, (batch_size, 1))
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.shape == (batch_size, 1, vocab_size)
        assert torch.isfinite(output).all()
    
    def test_parameter_count(self, config):
        """Test that model has reasonable parameter count."""
        model = HOPE(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])