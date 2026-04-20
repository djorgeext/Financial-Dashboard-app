"""
Model service for loading and running PyTorch neural network predictions.
Handles model loading, inference, and error management.
"""
import os
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import joblib
from pathlib import Path

from .utils import LSTMMixedModel

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing PyTorch model inference."""
    
    def __init__(self, model_dir: str = "models", device: Optional[str] = None):
        """
        Initialize the ModelService.
        
        Args:
            model_dir: Directory containing pre-trained model files
            device: torch device ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.models = {}
        self.scalers = {}
        logger.info(f"ModelService initialized. Device: {self.device}")
    
    def load_model(
        self, 
        model_name: str,
        num_features: int = 17,
        lstm_hidden: int = 128,
        num_classes: int = 3
    ) -> bool:
        """
        Load a PyTorch model from disk.
        
        Args:
            model_name: Name of the model file (without .pth extension)
            num_features: Number of input features
            lstm_hidden: LSTM hidden dimension
            num_classes: Number of output classes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = self.model_dir / f"{model_name}.pth"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Initialize model architecture
            model = LSTMMixedModel(
                num_features=num_features,
                lstm_hidden=lstm_hidden,
                num_classes=num_classes
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            logger.info(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def load_scaler(self, scaler_name: str) -> bool:
        """
        Load a fitted RobustScaler from disk.
        
        Args:
            scaler_name: Name of the scaler file (without .pkl extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            scaler_path = self.model_dir / f"{scaler_name}.pkl"
            
            if not scaler_path.exists():
                logger.warning(f"Scaler file not found: {scaler_path}")
                return False
            
            scaler = joblib.load(scaler_path)
            self.scalers[scaler_name] = scaler
            logger.info(f"Scaler loaded successfully: {scaler_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading scaler {scaler_name}: {str(e)}")
            return False
    
    def predict(
        self,
        model_name: str,
        features: np.ndarray
    ) -> Dict[str, any]:
        """
        Run prediction on input features.
        
        Args:
            model_name: Key of model to use (must be pre-loaded)
            features: Input features with shape (batch_size, sequence_length, num_features)
                     or (sequence_length, num_features) for single sample
            
        Returns:
            Dictionary with keys:
                - logits: Raw model output
                - probabilities: Softmax probabilities
                - predicted_class: Argmax class (0=down, 1=neutral, 2=up)
                - confidence: Max probability value
                - error: Error message if prediction failed
        """
        try:
            if model_name not in self.models:
                return {"error": f"Model {model_name} not loaded"}
            
            model = self.models[model_name]
            
            # Handle single sample vs batch
            if isinstance(features, np.ndarray):
                if features.ndim == 2:
                    # Single sample: add batch dimension
                    features = np.expand_dims(features, axis=0)
                features = torch.from_numpy(features).float().to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits, attention_weights = model(features)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
            
            # Convert to numpy
            result = {
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probabilities.cpu().numpy().tolist(),
                "predicted_class": predicted_class.cpu().numpy().tolist(),
                "confidence": confidence.cpu().numpy().tolist(),
                "class_names": ["down", "neutral", "up"]
            }
            
            logger.info(f"Prediction successful for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"error": str(e)}
    
    def get_model_status(self) -> Dict[str, bool]:
        """
        Get status of all loaded models.
        
        Returns:
            Dictionary mapping model names to loaded status
        """
        return {name: True for name in self.models.keys()}
    
    def model_ready(self, model_name: str) -> bool:
        """Check if a specific model is loaded and ready."""
        return model_name in self.models


def create_model_service(model_dir: str = "models") -> ModelService:
    """Factory function to create and initialize ModelService."""
    service = ModelService(model_dir=model_dir)
    return service
