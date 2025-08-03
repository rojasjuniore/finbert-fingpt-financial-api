"""
FinBERT service for financial sentiment analysis
"""

import time
import torch
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from loguru import logger
import numpy as np

from ..core.config import get_settings
from ..models.responses import SentimentScore

settings = get_settings()


class FinBERTService:
    """Service for financial sentiment analysis using FinBERT model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = None
        self.model_loaded = False
        self.load_time = None
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Sentiment labels mapping
        self.label_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral", 
            "LABEL_2": "positive"
        }
    
    async def load_model(self) -> bool:
        """Load the FinBERT model and tokenizer"""
        try:
            start_time = time.time()
            logger.info(f"Loading FinBERT model: {settings.model_name}")
            
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                use_auth_token=settings.hf_token,
                cache_dir=settings.transformers_cache
            )
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.model_name,
                use_auth_token=settings.hf_token,
                cache_dir=settings.transformers_cache
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            self.load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.success(f"Model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    async def analyze_sentiment(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = False,
        batch_size: Optional[int] = None
    ) -> Union[SentimentScore, List[SentimentScore]]:
        """
        Analyze sentiment of text or list of texts
        
        Args:
            text: Single text or list of texts to analyze
            return_probabilities: Whether to return probability scores
            batch_size: Batch size for processing multiple texts
            
        Returns:
            SentimentScore or list of SentimentScore objects
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Handle single text
        if isinstance(text, str):
            result = await self._analyze_single_text(text, return_probabilities)
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Update statistics
            self.inference_count += 1
            self.total_inference_time += processing_time
            
            return result
        
        # Handle multiple texts
        elif isinstance(text, list):
            results = await self._analyze_multiple_texts(
                text, return_probabilities, batch_size or settings.batch_size
            )
            processing_time = time.time() - start_time
            
            # Update statistics
            self.inference_count += len(text)
            self.total_inference_time += processing_time
            
            return results
        
        else:
            raise ValueError("Text must be string or list of strings")
    
    async def _analyze_single_text(
        self, 
        text: str, 
        return_probabilities: bool
    ) -> SentimentScore:
        """Analyze sentiment of a single text"""
        try:
            # Truncate text if too long
            if len(text) > settings.max_sequence_length * 4:  # Rough character estimate
                text = text[:settings.max_sequence_length * 4]
                logger.warning("Text truncated due to length")
            
            # Run inference
            results = self.pipeline(text)
            
            # Extract results
            scores = {self.label_mapping.get(item['label'], item['label']): item['score'] 
                     for item in results[0]}
            
            # Get prediction
            predicted_label = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[predicted_label]
            
            # Create response
            sentiment_score = SentimentScore(
                text=text,
                sentiment=predicted_label,
                confidence=confidence
            )
            
            if return_probabilities:
                sentiment_score.probabilities = scores
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise RuntimeError(f"Sentiment analysis failed: {str(e)}")
    
    async def _analyze_multiple_texts(
        self,
        texts: List[str],
        return_probabilities: bool,
        batch_size: int
    ) -> List[SentimentScore]:
        """Analyze sentiment of multiple texts in batches"""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Truncate texts if too long
                processed_batch = []
                for text in batch:
                    if len(text) > settings.max_sequence_length * 4:
                        text = text[:settings.max_sequence_length * 4]
                    processed_batch.append(text)
                
                # Run batch inference
                batch_results = self.pipeline(processed_batch)
                
                # Process results
                for j, (text, result) in enumerate(zip(batch, batch_results)):
                    scores = {self.label_mapping.get(item['label'], item['label']): item['score'] 
                             for item in result}
                    
                    predicted_label = max(scores.keys(), key=lambda k: scores[k])
                    confidence = scores[predicted_label]
                    
                    sentiment_score = SentimentScore(
                        text=text,
                        sentiment=predicted_label,
                        confidence=confidence
                    )
                    
                    if return_probabilities:
                        sentiment_score.probabilities = scores
                    
                    results.append(sentiment_score)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add error results for this batch
                for text in batch:
                    results.append(SentimentScore(
                        text=text,
                        sentiment="error",
                        confidence=0.0
                    ))
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {"model_loaded": False}
        
        return {
            "model_loaded": True,
            "model_name": settings.model_name,
            "device": str(self.device),
            "load_time": self.load_time,
            "max_sequence_length": settings.max_sequence_length,
            "batch_size": settings.batch_size,
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": (
                self.total_inference_time / self.inference_count 
                if self.inference_count > 0 else 0
            )
        }
    
    async def health_check(self, deep_check: bool = False) -> Dict:
        """Perform health check of the service"""
        health_info = {
            "model_loaded": self.model_loaded,
            "device": str(self.device) if self.device else None,
            "load_time": self.load_time
        }
        
        if deep_check and self.model_loaded:
            try:
                # Test inference with sample text
                test_text = "The company reported strong quarterly earnings with significant growth."
                start_time = time.time()
                result = await self._analyze_single_text(test_text, False)
                inference_time = time.time() - start_time
                
                health_info.update({
                    "test_inference_successful": True,
                    "test_inference_time": inference_time,
                    "test_result": {
                        "sentiment": result.sentiment,
                        "confidence": result.confidence
                    }
                })
                
            except Exception as e:
                health_info.update({
                    "test_inference_successful": False,
                    "test_error": str(e)
                })
        
        return health_info


# Global service instance
finbert_service = FinBERTService()