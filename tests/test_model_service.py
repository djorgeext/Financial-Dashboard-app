"""
Unit tests for model_service module.
Tests model loading, inference, and error handling.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import os

from backend.model_service import ModelService, create_model_service
from backend.utils import LSTMMixedModel


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for test models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_model_service(temp_model_dir):
    """Create a ModelService instance for testing."""
    return ModelService(model_dir=temp_model_dir, device='cpu')


@pytest.fixture
def sample_features():
    """Generate sample features for testing."""
    # Shape: (batch_size=1, sequence_length=40, num_features=17)
    return np.random.randn(1, 40, 17).astype(np.float32)


class TestModelLoading:
    """Tests for model loading functionality."""
    
    def test_model_not_found(self, test_model_service):
        """Test handling of missing model file."""
        result = test_model_service.load_model('nonexistent_model')
        assert result is False
    
    def test_model_load_success(self, test_model_service, temp_model_dir):
        """Test successful model loading."""
        # Create a dummy model
        model = LSTMMixedModel(num_features=17)
        model_path = Path(temp_model_dir) / 'test_model.pth'
        torch.save(model.state_dict(), model_path)
        
        # Load it
        result = test_model_service.load_model('test_model')
        assert result is True
        assert 'test_model' in test_model_service.models
    
    def test_load_scaler_not_found(self, test_model_service):
        """Test handling of missing scaler file."""
        result = test_model_service.load_scaler('nonexistent_scaler')
        assert result is False
    
    def test_model_ready_check(self, test_model_service, temp_model_dir):
        """Test model readiness check."""
        assert test_model_service.model_ready('nonexistent') is False
        
        # Create and load a model
        model = LSTMMixedModel(num_features=17)
        model_path = Path(temp_model_dir) / 'ready_test.pth'
        torch.save(model.state_dict(), model_path)
        
        test_model_service.load_model('ready_test')
        assert test_model_service.model_ready('ready_test') is True


class TestInference:
    """Tests for model inference."""
    
    def test_predict_model_not_loaded(self, test_model_service, sample_features):
        """Test prediction with unloaded model."""
        result = test_model_service.predict('nonexistent_model', sample_features)
        assert 'error' in result
    
    def test_predict_success(self, test_model_service, temp_model_dir, sample_features):
        """Test successful prediction."""
        # Create and load a model
        model = LSTMMixedModel(num_features=17)
        model_path = Path(temp_model_dir) / 'inference_test.pth'
        torch.save(model.state_dict(), model_path)
        
        test_model_service.load_model('inference_test')
        result = test_model_service.predict('inference_test', sample_features)
        
        assert 'error' not in result
        assert 'logits' in result
        assert 'probabilities' in result
        assert 'predicted_class' in result
        assert 'confidence' in result
    
    def test_predict_single_sample(self, test_model_service, temp_model_dir):
        """Test prediction with single sample (2D input)."""
        # Create and load a model
        model = LSTMMixedModel(num_features=17)
        model_path = Path(temp_model_dir) / 'single_sample_test.pth'
        torch.save(model.state_dict(), model_path)
        
        test_model_service.load_model('single_sample_test')
        
        # Single sample: (sequence_length, num_features)
        features = np.random.randn(40, 17).astype(np.float32)
        result = test_model_service.predict('single_sample_test', features)
        
        assert 'error' not in result
        assert len(result['predicted_class']) == 1
    
    def test_predict_output_structure(self, test_model_service, temp_model_dir):
        """Test prediction output structure."""
        model = LSTMMixedModel(num_features=17)
        model_path = Path(temp_model_dir) / 'structure_test.pth'
        torch.save(model.state_dict(), model_path)
        
        test_model_service.load_model('structure_test')
        features = np.random.randn(2, 40, 17).astype(np.float32)
        result = test_model_service.predict('structure_test', features)
        
        # Verify structure
        assert isinstance(result['logits'], list)
        assert isinstance(result['probabilities'], list)
        assert isinstance(result['predicted_class'], list)
        assert isinstance(result['confidence'], list)
        assert 'class_names' in result
        assert len(result['class_names']) == 3
    
    def test_probabilities_sum_to_one(self, test_model_service, temp_model_dir):
        """Test that predicted probabilities sum to 1."""
        model = LSTMMixedModel(num_features=17)
        model_path = Path(temp_model_dir) / 'prob_test.pth'
        torch.save(model.state_dict(), model_path)
        
        test_model_service.load_model('prob_test')
        features = np.random.randn(1, 40, 17).astype(np.float32)
        result = test_model_service.predict('prob_test', features)
        
        probs = result['probabilities'][0]
        total = sum(probs)
        assert abs(total - 1.0) < 1e-5  # Allow small floating point error


class TestModelStatus:
    """Tests for model status tracking."""
    
    def test_get_model_status_empty(self, test_model_service):
        """Test model status when no models loaded."""
        status = test_model_service.get_model_status()
        assert status == {}
    
    def test_get_model_status_multiple(self, test_model_service, temp_model_dir):
        """Test model status with multiple models."""
        # Load multiple models
        for i in range(3):
            model = LSTMMixedModel(num_features=17)
            model_path = Path(temp_model_dir) / f'model_{i}.pth'
            torch.save(model.state_dict(), model_path)
            test_model_service.load_model(f'model_{i}')
        
        status = test_model_service.get_model_status()
        assert len(status) == 3
        assert all(v for v in status.values())  # All True


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_model_service(self, temp_model_dir):
        """Test ModelService creation via factory."""
        service = create_model_service(model_dir=temp_model_dir)
        assert service is not None
        assert isinstance(service, ModelService)
