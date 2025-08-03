"""
FinGPT service for financial text generation and analysis
"""

import time
import torch
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from loguru import logger
import asyncio

from ..core.config import get_settings
from ..models.responses import FinGPTResponse
from .finnhub_service import finnhub_service

settings = get_settings()


class FinGPTService:
    """Service for financial text generation using FinGPT model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = None
        self.model_loaded = False
        self.load_time = None
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # FinGPT model configuration
        self.model_name = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
        self.backup_model = "microsoft/DialoGPT-medium"  # Fallback if FinGPT not available
    
    async def load_model(self) -> bool:
        """Load the FinGPT model and tokenizer"""
        try:
            start_time = time.time()
            logger.info(f"Loading FinGPT model: {self.model_name}")
            
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            try:
                # Try to load FinGPT first
                logger.info("Attempting to load FinGPT model...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=settings.hf_token,
                    cache_dir=settings.hf_home
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=settings.hf_token,
                    cache_dir=settings.hf_home,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                
                logger.success(f"Successfully loaded FinGPT model: {self.model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load FinGPT model: {str(e)}")
                logger.info(f"Falling back to backup model: {self.backup_model}")
                
                # Fallback to backup model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.backup_model,
                    cache_dir=settings.hf_home
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.backup_model,
                    cache_dir=settings.hf_home,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                
                self.model_name = self.backup_model
                logger.success(f"Successfully loaded backup model: {self.backup_model}")
            
            # Move model to device if not using device_map
            if self.device.type == "cpu":
                self.model.to(self.device)
            
            self.model.eval()
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            self.load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.success(f"FinGPT model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FinGPT model: {str(e)}")
            self.model_loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    async def generate_text(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> Union[FinGPTResponse, List[FinGPTResponse]]:
        """
        Generate financial text using FinGPT model
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to generate
            
        Returns:
            FinGPTResponse or list of FinGPTResponse objects
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Prepare generation parameters
            generation_params = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            # Generate text
            logger.debug(f"Generating text with prompt: {prompt[:100]}...")
            results = self.pipeline(prompt, **generation_params)
            
            processing_time = time.time() - start_time
            
            # Process results
            responses = []
            for i, result in enumerate(results):
                generated_text = result['generated_text']
                
                response = FinGPTResponse(
                    prompt=prompt,
                    generated_text=generated_text,
                    model_name=self.model_name,
                    processing_time=processing_time / len(results),
                    generation_params={
                        "max_length": max_length,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "do_sample": do_sample
                    }
                )
                responses.append(response)
            
            # Update statistics
            self.inference_count += len(results)
            self.total_inference_time += processing_time
            
            # Return single response or list
            if num_return_sequences == 1:
                return responses[0]
            else:
                return responses
                
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    async def analyze_financial_text(
        self,
        text: str,
        analysis_type: str = "general",
        symbol: Optional[str] = None
    ) -> Dict:
        """
        Analyze financial text using FinGPT with optional Finnhub data enhancement
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (general, sentiment, forecast, risk)
            symbol: Optional stock symbol for enhanced analysis with Finnhub data
            
        Returns:
            Analysis results dictionary
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get enhanced context from Finnhub if symbol provided
        finnhub_context = ""
        finnhub_data = None
        if symbol and finnhub_service.is_available():
            try:
                logger.info(f"Fetching Finnhub data for enhanced analysis: {symbol}")
                finnhub_data = await finnhub_service.get_comprehensive_data(symbol)
                
                if finnhub_data.get('available'):
                    insights = finnhub_service.extract_key_insights(finnhub_data)
                    if insights:
                        finnhub_context = f"\\n\\nReal-time market data for {symbol}:\\n" + "\\n".join(f"• {insight}" for insight in insights[:8])
                        logger.info(f"Enhanced analysis with {len(insights)} Finnhub insights")
            except Exception as e:
                logger.warning(f"Could not fetch Finnhub data for {symbol}: {str(e)}")
        
        # Create analysis prompt based on type with enhanced context
        base_prompts = {
            "general": f"Analyze the following financial text and provide comprehensive insights:{finnhub_context}\\n\\nText to analyze: {text}\\n\\nAnalysis:",
            "sentiment": f"Analyze the sentiment of this financial text and explain the reasoning with market context:{finnhub_context}\\n\\nText: {text}\\n\\nSentiment Analysis:",
            "forecast": f"Based on this financial information and current market data, provide a detailed forecast:{finnhub_context}\\n\\nText: {text}\\n\\nMarket Forecast:",
            "risk": f"Identify and analyze financial risks with current market context:{finnhub_context}\\n\\nText: {text}\\n\\nRisk Analysis:"
        }
        
        prompt = base_prompts.get(analysis_type, base_prompts["general"])
        
        try:
            response = await self.generate_text(
                prompt=prompt,
                max_length=400 if finnhub_context else 300,  # Longer response for enhanced analysis
                temperature=0.3,  # Lower temperature for more focused analysis
                top_p=0.8,
                do_sample=True
            )
            
            result = {
                "analysis_type": analysis_type,
                "original_text": text,
                "analysis": response.generated_text,
                "model_used": self.model_name,
                "processing_time": response.processing_time,
                "enhanced_with_finnhub": bool(finnhub_context),
                "symbol": symbol.upper() if symbol else None
            }
            
            # Add key insights extracted from analysis
            if finnhub_data and finnhub_data.get('available'):
                result["market_data_insights"] = finnhub_service.extract_key_insights(finnhub_data)
                result["data_timestamp"] = finnhub_data.get('timestamp')
            
            return result
            
        except Exception as e:
            logger.error(f"Error in financial text analysis: {str(e)}")
            raise RuntimeError(f"Financial analysis failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {"model_loaded": False}
        
        return {
            "model_loaded": True,
            "model_name": self.model_name,
            "device": str(self.device),
            "load_time": self.load_time,
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
            "load_time": self.load_time,
            "model_name": self.model_name if self.model_loaded else None
        }
        
        if deep_check and self.model_loaded:
            try:
                # Test generation with sample prompt
                test_prompt = "The market outlook for technology stocks"
                start_time = time.time()
                result = await self.generate_text(
                    prompt=test_prompt,
                    max_length=50,
                    temperature=0.7
                )
                inference_time = time.time() - start_time
                
                health_info.update({
                    "test_generation_successful": True,
                    "test_generation_time": inference_time,
                    "test_result": {
                        "prompt": test_prompt,
                        "generated_length": len(result.generated_text)
                    }
                })
                
            except Exception as e:
                health_info.update({
                    "test_generation_successful": False,
                    "test_error": str(e)
                })
        
        return health_info


# Global service instance
fingpt_service = FinGPTService()