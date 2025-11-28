import pytest
import torch
from torch.utils.data import DataLoader
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add src to path so we can import the TextDataset
# sys.path.append(str(Path(__file__).parent.parent / "src"))
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.utils.data_loader import TextDataset, get_data_loader


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token = "<eos>"
    
    # Mock the tokenizer call to return realistic token IDs
    def mock_tokenize(text, return_tensors=None, truncation=None, max_length=None):
        # Simple mock: convert text to token IDs (one char = one token for simplicity)
        # Add some realistic variation
        token_ids = [i % 1000 for i in range(len(text))]  # Mock token IDs
        return {"input_ids": torch.tensor([token_ids])}
    
    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize
    
    return tokenizer


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "This is a sample text for testing the dataset.",
        "Another piece of text to verify functionality.",
        "Third text sample with different content.",
        "Final text to ensure we have enough data for testing."
    ]


def test_text_dataset_initialization(mock_tokenizer, sample_texts):
    """Test that TextDataset initializes correctly."""
    max_seq_len = 32
    
    dataset = TextDataset(sample_texts, mock_tokenizer, max_seq_len)
    
    # Should have tokenizer and max_seq_len set
    assert dataset.tokenizer == mock_tokenizer
    assert dataset.max_seq_len == max_seq_len
    
    # Should have encodings tensor
    assert hasattr(dataset, 'encodings')
    assert isinstance(dataset.encodings, torch.Tensor)


def test_text_dataset_length(mock_tokenizer, sample_texts):
    """Test that dataset length is calculated correctly."""
    max_seq_len = 10
    
    dataset = TextDataset(sample_texts, mock_tokenizer, max_seq_len)
    
    # Length should be (total_tokens - 1) // max_seq_len
    expected_length = (len(dataset.encodings) - 1) // max_seq_len
    assert len(dataset) == expected_length
    assert len(dataset) >= 0


def test_text_dataset_getitem(mock_tokenizer, sample_texts):
    """Test that dataset returns correct tensor chunks."""
    max_seq_len = 8
    
    dataset = TextDataset(sample_texts, mock_tokenizer, max_seq_len)
    
    if len(dataset) > 0:
        # Get first item
        item = dataset[0]
        
        # Should return a tensor of correct length
        assert isinstance(item, torch.Tensor)
        assert len(item) == max_seq_len
        
        # Should be within valid range for second item if it exists
        if len(dataset) > 1:
            item2 = dataset[1]
            assert isinstance(item2, torch.Tensor)
            assert len(item2) == max_seq_len


def test_text_dataset_different_sequence_lengths(mock_tokenizer, sample_texts):
    """Test dataset with different sequence lengths."""
    for seq_len in [4, 16, 32]:
        dataset = TextDataset(sample_texts, mock_tokenizer, seq_len)
        
        assert dataset.max_seq_len == seq_len
        
        if len(dataset) > 0:
            item = dataset[0]
            assert len(item) == seq_len


def test_text_dataset_edge_cases(mock_tokenizer):
    """Test edge cases like empty texts or single character."""
    # Empty text list
    dataset = TextDataset([], mock_tokenizer, 10)
    assert len(dataset) >= 0
    
    # Single character text
    single_char_texts = ["a"]
    dataset = TextDataset(single_char_texts, mock_tokenizer, 5)
    assert len(dataset) >= 0
    
    # Very long sequence length
    dataset = TextDataset(["short text"], mock_tokenizer, 1000)
    # Should handle gracefully without errors


def test_text_dataset_consistency(mock_tokenizer, sample_texts):
    """Test that dataset returns consistent results."""
    max_seq_len = 12
    
    dataset = TextDataset(sample_texts, mock_tokenizer, max_seq_len)
    
    if len(dataset) > 0:
        # Same index should return same result
        item1 = dataset[0]
        item2 = dataset[0]
        
        assert torch.equal(item1, item2)


@pytest.mark.parametrize("max_seq_len", [1, 8, 16, 32])
def test_text_dataset_parametrized_seq_lengths(mock_tokenizer, sample_texts, max_seq_len):
    """Test dataset with various sequence lengths using parametrize."""
    dataset = TextDataset(sample_texts, mock_tokenizer, max_seq_len)
    
    assert dataset.max_seq_len == max_seq_len
    assert len(dataset) >= 0
    
    if len(dataset) > 0:
        item = dataset[0]
        assert len(item) == max_seq_len
        assert item.dtype == torch.long or item.dtype == torch.int


def test_text_dataset_with_dataloader(mock_tokenizer, sample_texts):
    """Test that TextDataset works with PyTorch DataLoader."""
    max_seq_len = 16
    batch_size = 2
    
    dataset = TextDataset(sample_texts, mock_tokenizer, max_seq_len)
    
    if len(dataset) >= batch_size:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Get first batch
        batch = next(iter(dataloader))
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape[0] == batch_size  # Batch dimension
        assert batch.shape[1] == max_seq_len  # Sequence dimension


def test_text_dataset_memory_efficiency(mock_tokenizer):
    """Test that dataset doesn't consume excessive memory with larger inputs."""
    # Create a moderately sized input
    large_texts = ["This is a test text. " * 100] * 10  # Repeat to make it larger
    max_seq_len = 64
    
    # Should initialize without memory errors
    dataset = TextDataset(large_texts, mock_tokenizer, max_seq_len)
    
    assert len(dataset) >= 0
    
    if len(dataset) > 0:
        # Should be able to access items without memory issues
        item = dataset[0]
        assert len(item) == max_seq_len