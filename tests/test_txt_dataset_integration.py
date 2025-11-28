import pytest
import torch
from unittest.mock import patch, Mock
import sys
from pathlib import Path

# Add src to path
# sys.path.append(str(Path(__file__).parent.parent / "src"))
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.utils.data_loader import get_data_loader


@pytest.fixture
def mock_dataset_response():
    """Mock the datasets library response."""
    return [
        {"text": "First sample text for testing"},
        {"text": "Second sample with different content"},
        {"text": ""},  # Empty text that should be filtered
        {"text": "   "},  # Whitespace-only that should be filtered
        {"text": "Final valid text sample"}
    ]


@patch('src.utils.data_loader.load_dataset')
@patch('src.utils.data_loader.AutoTokenizer.from_pretrained')
def test_get_data_loader_smoke(mock_tokenizer_class, mock_load_dataset, mock_dataset_response):
    """Smoke test for get_data_loader function."""
    # Mock the dataset loading
    mock_load_dataset.return_value = mock_dataset_response
    
    # Mock the tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.pad_token = "<eos>"
    
    # Mock tokenizer call
    def mock_tokenize(text, return_tensors=None, truncation=None, max_length=None):
        token_ids = [i % 100 for i in range(min(50, len(text)))]  # Limit size
        return {"input_ids": torch.tensor([token_ids])}
    
    mock_tokenizer.side_effect = mock_tokenize
    mock_tokenizer.__call__ = mock_tokenize
    mock_tokenizer_class.return_value = mock_tokenizer
    
    # Test the function
    batch_size = 2
    max_seq_len = 16
    
    dataloader = get_data_loader(
        dataset_name='wikitext-103',
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        split='train'
    )
    
    # Should return a DataLoader
    assert hasattr(dataloader, '__iter__')
    assert hasattr(dataloader, 'batch_size')
    assert dataloader.batch_size == batch_size
    
    # Verify mocks were called
    mock_load_dataset.assert_called_once()
    mock_tokenizer_class.assert_called_once_with("gpt2")


@patch('src.utils.data_loader.load_dataset')
@patch('src.utils.data_loader.AutoTokenizer.from_pretrained')
def test_get_data_loader_different_datasets(mock_tokenizer_class, mock_load_dataset, mock_dataset_response):
    """Test get_data_loader with different dataset names."""
    mock_load_dataset.return_value = mock_dataset_response
    
    # Mock tokenizer setup
    mock_tokenizer = Mock()
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.pad_token = "<eos>"
    
    def mock_tokenize(text, return_tensors=None, truncation=None, max_length=None):
        # Return enough tokens to ensure dataset has at least 1 item
        # With max_seq_len=16, we need at least 17 tokens to get length 1
        return {"input_ids": torch.tensor([[i for i in range(20)]])}
    
    
    mock_tokenizer.side_effect = mock_tokenize
    mock_tokenizer.__call__ = mock_tokenize
    mock_tokenizer_class.return_value = mock_tokenizer
    
    # Test with wikitext-103
    dataloader1 = get_data_loader('wikitext-103', 2, 16, 'train')
    assert dataloader1 is not None
    
    # Test with fallback dataset (anything other than wikitext-103)
    dataloader2 = get_data_loader('other-dataset', 2, 16, 'train')
    assert dataloader2 is not None
    
    # Should have called load_dataset twice
    assert mock_load_dataset.call_count == 2


@patch('src.utils.data_loader.load_dataset')
@patch('src.utils.data_loader.AutoTokenizer.from_pretrained')
def test_data_loader_filters_empty_texts(mock_tokenizer_class, mock_load_dataset):
    """Test that empty texts are properly filtered."""
    # Dataset with many empty/whitespace texts
    mock_data = [
        {"text": "Valid text"},
        {"text": ""},
        {"text": "   "},
        {"text": "\n\n"},
        {"text": "Another valid text"},
        {"text": "\t  \n  "},
    ]
    mock_load_dataset.return_value = mock_data
    
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.pad_token = "<eos>"
    
    # Track what texts were passed to tokenizer
    received_texts = []
    
    def mock_tokenize(text, return_tensors=None, truncation=None, max_length=None):
        received_texts.append(text)
        # Return enough tokens so dataset has at least 1 item
        # With max_seq_len=8, we need at least 9 tokens
        return {"input_ids": torch.tensor([[i for i in range(10)]])}
    
    mock_tokenizer.side_effect = mock_tokenize
    mock_tokenizer.__call__ = mock_tokenize
    mock_tokenizer_class.return_value = mock_tokenizer
    
    # Create data loader
    get_data_loader('test', 1, 8, 'train')
    
    # Should have filtered empty texts - only valid texts should remain
    # The join should only include "Valid text" and "Another valid text"
    assert len(received_texts) == 1  # One call with joined text
    joined_text = received_texts[0]
    assert "Valid text" in joined_text
    assert "Another valid text" in joined_text
    # Should not contain only whitespace