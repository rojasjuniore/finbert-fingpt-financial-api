"""
Load tests for FinBERT API using Locust.
"""
from locust import HttpUser, task, between
import json
import random
import time


class FinBERTAPIUser(HttpUser):
    """Simulated user for FinBERT API load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts running."""
        self.financial_texts = [
            # Positive sentiment texts
            "Apple Inc. reported record quarterly earnings, beating analyst expectations by 15%.",
            "Tesla stock surges 20% after announcing breakthrough in battery technology.",
            "Microsoft announces strong cloud revenue growth and raises guidance for next quarter.",
            "Amazon Web Services shows continued growth with 32% increase in quarterly revenue.",
            "Google parent Alphabet beats earnings estimates with strong advertising revenue.",
            
            # Negative sentiment texts
            "Company faces significant challenges due to supply chain disruptions.",
            "Quarterly losses exceed expectations, causing concern among investors.",
            "Regulatory investigation threatens future business operations and profitability.",
            "Stock price plummets 25% following disappointing earnings announcement.",
            "Credit rating downgraded due to mounting debt and declining revenues.",
            
            # Neutral sentiment texts
            "The company will hold its annual shareholders meeting next month.",
            "Board of directors announces retirement of long-serving CEO.",
            "Company files routine regulatory documents with the SEC.",
            "New factory construction project remains on schedule for completion.",
            "Quarterly dividend payment date announced for registered shareholders.",
            
            # Mixed/complex texts
            "Despite strong revenue growth, the company reported lower margins due to increased costs.",
            "Acquisition completed successfully, though integration challenges remain ahead.",
            "Product launch receives positive reviews but faces competitive market pressures.",
            "Strong domestic sales offset by weaker international performance this quarter.",
            "Innovation investments show promise but have not yet translated to revenue gains."
        ]
        
        self.batch_texts = [
            self.financial_texts[:3],
            self.financial_texts[3:6],
            self.financial_texts[6:9],
            self.financial_texts[9:12],
        ]
    
    @task(10)
    def analyze_sentiment(self):
        """Test single sentiment analysis endpoint."""
        text = random.choice(self.financial_texts)
        payload = {"text": text}
        
        with self.client.post("/analyze", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Validate response structure
                    required_fields = ["text", "sentiment", "processing_time", "model_info"]
                    if all(field in data for field in required_fields):
                        sentiment = data["sentiment"]
                        if (sentiment.get("label") in ["positive", "negative", "neutral"] and
                            0.0 <= sentiment.get("score", -1) <= 1.0 and
                            0.0 <= sentiment.get("confidence", -1) <= 1.0):
                            response.success()
                        else:
                            response.failure("Invalid sentiment data structure")
                    else:
                        response.failure("Missing required fields in response")
                        
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 400:
                response.success()  # Expected for some invalid inputs
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(3)
    def batch_analyze_sentiment(self):
        """Test batch sentiment analysis endpoint."""
        texts = random.choice(self.batch_texts)
        payload = {"texts": texts}
        
        with self.client.post("/batch-analyze", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Validate batch response structure
                    required_fields = ["results", "batch_size", "processing_time"]
                    if all(field in data for field in required_fields):
                        if (data["batch_size"] == len(texts) and
                            len(data["results"]) == len(texts)):
                            
                            # Validate individual results
                            all_valid = True
                            for result in data["results"]:
                                if not all(field in result for field in ["text", "sentiment", "processing_time"]):
                                    all_valid = False
                                    break
                                sentiment = result["sentiment"]
                                if not (sentiment.get("label") in ["positive", "negative", "neutral"] and
                                       0.0 <= sentiment.get("score", -1) <= 1.0):
                                    all_valid = False
                                    break
                            
                            if all_valid:
                                response.success()
                            else:
                                response.failure("Invalid individual result structure")
                        else:
                            response.failure("Batch size mismatch")
                    else:
                        response.failure("Missing required fields in batch response")
                        
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure("Health check returned unhealthy status")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Health check failed with status: {response.status_code}")
    
    @task(1)
    def model_info(self):
        """Test model info endpoint."""
        with self.client.get("/model-info", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    required_fields = ["model_name", "max_sequence_length"]
                    if all(field in data for field in required_fields):
                        response.success()
                    else:
                        response.failure("Missing required fields in model info")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Model info failed with status: {response.status_code}")
    
    @task(1)
    def metrics_endpoint(self):
        """Test metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # Metrics should be in Prometheus format
                if "finbert_requests_total" in response.text:
                    response.success()
                else:
                    response.failure("Metrics endpoint missing expected content")
            else:
                response.failure(f"Metrics endpoint failed with status: {response.status_code}")


class HighLoadUser(HttpUser):
    """High-load user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very short wait time for stress testing
    
    def on_start(self):
        """Initialize with shorter texts for faster processing."""
        self.short_texts = [
            "Strong earnings",
            "Stock declined",
            "Market neutral",
            "Revenue growth",
            "Profit warning",
            "Guidance raised",
            "Costs increased",
            "Sales improved",
            "Margins compressed",
            "Outlook positive"
        ]
    
    @task
    def rapid_analysis(self):
        """Rapid sentiment analysis for stress testing."""
        text = random.choice(self.short_texts)
        payload = {"text": text}
        
        with self.client.post("/analyze", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code in [429, 503]:  # Rate limited or service unavailable
                response.success()  # Expected under high load
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class ErrorTestUser(HttpUser):
    """User that tests error handling under load."""
    
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Initialize with various error-inducing inputs."""
        self.error_inputs = [
            "",  # Empty string
            " " * 10000,  # Very long string
            None,  # None value
            123,  # Wrong type
            {"invalid": "structure"},  # Wrong structure
            "Special chars: !@#$%^&*()",
            "Unicode: émojis 📈 📉 💰"
        ]
    
    @task(5)
    def test_error_handling(self):
        """Test error handling with various invalid inputs."""
        error_input = random.choice(self.error_inputs)
        
        try:
            payload = {"text": error_input}
            with self.client.post("/analyze", json=payload, catch_response=True) as response:
                if response.status_code in [200, 400, 422]:  # Expected responses
                    response.success()
                else:
                    response.failure(f"Unexpected error status: {response.status_code}")
        except Exception:
            # Some inputs might cause JSON serialization errors
            pass
    
    @task(3)
    def test_malformed_requests(self):
        """Test with malformed JSON requests."""
        malformed_payloads = [
            '{"text": invalid}',  # Invalid JSON
            '{"text":}',  # Incomplete JSON
            '{"wrong_field": "value"}',  # Wrong field name
            '{',  # Incomplete JSON
            '',  # Empty payload
        ]
        
        payload = random.choice(malformed_payloads)
        
        with self.client.post(
            "/analyze",
            data=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code in [400, 422]:  # Expected error responses
                response.success()
            else:
                response.failure(f"Unexpected status for malformed request: {response.status_code}")
    
    @task(2)
    def test_invalid_endpoints(self):
        """Test invalid endpoints."""
        invalid_endpoints = [
            "/invalid",
            "/analyze/invalid",
            "/batch-analyze/extra",
            "/../../../etc/passwd",
            "/admin",
            "/debug"
        ]
        
        endpoint = random.choice(invalid_endpoints)
        
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code in [404, 405]:  # Expected for invalid endpoints
                response.success()
            else:
                response.failure(f"Unexpected status for invalid endpoint: {response.status_code}")


class ConcurrencyTestUser(HttpUser):
    """User for testing concurrency limits."""
    
    wait_time = between(0, 0.1)  # Minimal wait time
    
    @task
    def concurrent_requests(self):
        """Make requests with minimal delay to test concurrency."""
        payload = {"text": f"Concurrent test at {time.time()}"}
        
        with self.client.post("/analyze", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code in [429, 503, 502, 504]:  # Various overload responses
                response.success()  # Expected under high concurrency
            else:
                response.failure(f"Unexpected status: {response.status_code}")


# Custom Locust test scenarios
def create_load_test_scenarios():
    """Create different load test scenarios."""
    
    scenarios = {
        "normal_load": {
            "user_classes": [FinBERTAPIUser],
            "users": 10,
            "spawn_rate": 2,
            "run_time": "5m"
        },
        
        "spike_load": {
            "user_classes": [FinBERTAPIUser, HighLoadUser],
            "users": 50,
            "spawn_rate": 10,
            "run_time": "2m"
        },
        
        "stress_test": {
            "user_classes": [HighLoadUser],
            "users": 100,
            "spawn_rate": 20,
            "run_time": "3m"
        },
        
        "error_handling": {
            "user_classes": [ErrorTestUser],
            "users": 20,
            "spawn_rate": 5,
            "run_time": "3m"
        },
        
        "concurrency_test": {
            "user_classes": [ConcurrencyTestUser],
            "users": 30,
            "spawn_rate": 15,
            "run_time": "2m"
        },
        
        "mixed_load": {
            "user_classes": [FinBERTAPIUser, HighLoadUser, ErrorTestUser],
            "users": 40,
            "spawn_rate": 8,
            "run_time": "10m"
        }
    }
    
    return scenarios


if __name__ == "__main__":
    """
    Run specific load test scenarios.
    
    Usage:
    locust -f locustfile.py --host=http://localhost:8000
    locust -f locustfile.py FinBERTAPIUser --host=http://localhost:8000
    locust -f locustfile.py HighLoadUser --host=http://localhost:8000 --users=100 --spawn-rate=20
    """
    pass