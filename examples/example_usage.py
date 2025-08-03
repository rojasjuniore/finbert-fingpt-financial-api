#!/usr/bin/env python3
"""
Example usage of the FinBERT API
"""

import requests
import json
import time
from typing import List, Dict, Any


class FinBERTClient:
    """Simple client for FinBERT API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/v1"
    
    def analyze_text(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """Analyze a single text"""
        response = requests.post(
            f"{self.api_url}/analyze",
            json={
                "text": text,
                "return_probabilities": return_probabilities
            }
        )
        response.raise_for_status()
        return response.json()
    
    def analyze_multiple(self, texts: List[str], return_probabilities: bool = False, batch_size: int = 32) -> Dict[str, Any]:
        """Analyze multiple texts"""
        response = requests.post(
            f"{self.api_url}/analyze",
            json={
                "text": texts,
                "return_probabilities": return_probabilities,
                "batch_size": batch_size
            }
        )
        response.raise_for_status()
        return response.json()
    
    def bulk_analyze(self, texts: List[str], return_probabilities: bool = False, batch_size: int = 32) -> Dict[str, Any]:
        """Bulk analyze texts with statistics"""
        response = requests.post(
            f"{self.api_url}/analyze/bulk",
            json={
                "texts": texts,
                "return_probabilities": return_probabilities,
                "batch_size": batch_size
            }
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self, deep_check: bool = False) -> Dict[str, Any]:
        """Check API health"""
        response = requests.get(
            f"{self.api_url}/health",
            params={"deep_check": deep_check}
        )
        response.raise_for_status()
        return response.json()
    
    def model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = requests.get(f"{self.api_url}/model/info")
        response.raise_for_status()
        return response.json()


def main():
    """Main example function"""
    
    # Initialize client
    client = FinBERTClient()
    
    # Check if API is available
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the FinBERT API is running at http://localhost:8000")
        return
    
    # Example 1: Single text analysis
    print("=== Example 1: Single Text Analysis ===")
    text = "The company reported strong quarterly earnings with significant growth in revenue."
    
    try:
        result = client.analyze_text(text, return_probabilities=True)
        data = result['data']
        
        print(f"Text: {text}")
        print(f"Sentiment: {data['sentiment']}")
        print(f"Confidence: {data['confidence']:.3f}")
        
        if 'probabilities' in data:
            print("Probabilities:")
            for sentiment, prob in data['probabilities'].items():
                print(f"  {sentiment}: {prob:.3f}")
        
        print(f"Processing time: {data.get('processing_time', 0):.3f}s")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 2: Multiple texts analysis
    print("=== Example 2: Multiple Texts Analysis ===")
    texts = [
        "The stock market reached new highs today with strong investor confidence.",
        "Economic uncertainty led to a significant drop in market values.",
        "The Federal Reserve maintained interest rates at current levels.",
        "Tech stocks showed mixed performance in today's trading session.",
        "Quarterly earnings reports exceeded analyst expectations across sectors."
    ]
    
    try:
        result = client.analyze_multiple(texts, return_probabilities=False)
        
        print(f"Analyzed {len(texts)} texts:")
        for i, item in enumerate(result['data']):
            print(f"{i+1}. Sentiment: {item['sentiment']:<10} Confidence: {item['confidence']:.3f}")
            print(f"   Text: {item['text'][:60]}...")
        
        print(f"Total processing time: {result.get('total_processing_time', 0):.3f}s")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 3: Bulk analysis with statistics
    print("=== Example 3: Bulk Analysis with Statistics ===")
    
    # Generate more sample texts
    sample_texts = [
        "Revenue growth exceeded expectations by 15% this quarter.",
        "The market crash caused significant losses for investors.",
        "Company stock remains stable despite market volatility.",
        "Strong earnings report boosted investor confidence significantly.",
        "Economic downturn led to reduced consumer spending patterns.",
        "New product launch received positive market reception.",
        "Regulatory changes may impact future profitability margins.",
        "Market analysts upgraded the stock to buy rating.",
        "Disappointing quarterly results led to stock price decline.",
        "The company announced a major acquisition deal today."
    ]
    
    try:
        start_time = time.time()
        result = client.bulk_analyze(sample_texts, return_probabilities=False, batch_size=5)
        end_time = time.time()
        
        print(f"Processed {result['processed_count']} texts in {result['total_processing_time']:.3f}s")
        print(f"Processing rate: {result['statistics']['processing_rate']:.2f} texts/second")
        print()
        
        print("Sentiment Distribution:")
        for sentiment, count in result['statistics']['sentiment_distribution'].items():
            percentage = (count / result['processed_count']) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        print(f"\nConfidence Statistics:")
        stats = result['statistics']
        print(f"  Average: {stats['average_confidence']:.3f}")
        print(f"  Min: {stats['min_confidence']:.3f}")
        print(f"  Max: {stats['max_confidence']:.3f}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 4: Model information
    print("=== Example 4: Model Information ===")
    try:
        info = client.model_info()
        
        print(f"Model: {info['model_name']}")
        print(f"Loaded: {info['model_loaded']}")
        print(f"Capabilities: {', '.join(info['capabilities'])}")
        
        if 'performance_stats' in info and info['performance_stats']:
            stats = info['performance_stats']
            print(f"\nPerformance Statistics:")
            print(f"  Total inferences: {stats['total_inferences']}")
            print(f"  Average inference time: {stats['average_inference_time']:.4f}s")
        
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 5: Error handling
    print("=== Example 5: Error Handling ===")
    try:
        # This should cause a validation error
        client.analyze_text("")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            error_data = e.response.json()
            print(f"Validation Error: {error_data['message']}")
            print(f"Details: {error_data.get('details', {})}")
        else:
            print(f"HTTP Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nExamples completed!")


if __name__ == "__main__":
    main()