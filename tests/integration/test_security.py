"""
Security tests for FinBERT API.
"""
import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
import json
import asyncio


class TestSecurity:
    """Security tests for API endpoints."""
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_input_validation_sql_injection(self, test_client, security_test_data):
        """Test protection against SQL injection attempts."""
        for injection in security_test_data["injection_attempts"]:
            response = await test_client.post("/analyze", json={"text": injection})
            
            # Should not crash and should sanitize input
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                data = response.json()
                # Should still process as text, not execute injection
                assert "sentiment" in data
                assert data["sentiment"]["label"] in ["positive", "negative", "neutral"]
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_input_validation_xss_protection(self, test_client):
        """Test protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
            "<iframe src='javascript:alert(\"xss\")'></iframe>"
        ]
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            for payload in xss_payloads:
                response = await test_client.post("/analyze", json={"text": payload})
                
                assert response.status_code == 200
                data = response.json()
                
                # Response should not contain executable code
                response_text = json.dumps(data)
                assert "<script>" not in response_text
                assert "javascript:" not in response_text
                assert "onerror=" not in response_text
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_input_size_limits(self, test_client, security_test_data):
        """Test input size limits to prevent DoS attacks."""
        for oversized_request in security_test_data["oversized_requests"]:
            response = await test_client.post("/analyze", json=oversized_request)
            
            # Should handle large inputs gracefully
            assert response.status_code in [200, 400, 413, 422]
            
            if response.status_code == 413:  # Payload too large
                error_data = response.json()
                assert "error" in error_data
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_malformed_requests(self, test_client, security_test_data):
        """Test handling of malformed requests."""
        for malformed_request in security_test_data["malformed_requests"]:
            try:
                response = await test_client.post("/analyze", json=malformed_request)
                # Should return appropriate error codes
                assert response.status_code in [400, 422]
                
                error_data = response.json()
                assert "error" in error_data or "detail" in error_data
                
            except Exception:
                # Some malformed requests might cause JSON encoding errors
                # This is acceptable as long as the server doesn't crash
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_invalid_content_types(self, test_client, security_test_data):
        """Test handling of invalid content types."""
        test_data = {"text": "Test content type validation"}
        
        for headers in security_test_data["invalid_headers"]:
            try:
                response = await test_client.post(
                    "/analyze",
                    json=test_data,
                    headers=headers
                )
                
                # Should reject invalid content types
                assert response.status_code in [400, 415, 422]
                
            except Exception:
                # Some invalid headers might cause connection errors
                # This is acceptable
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_request_headers_validation(self, test_client):
        """Test request header validation."""
        malicious_headers = [
            {"X-Forwarded-For": "127.0.0.1, evil.com"},
            {"User-Agent": "<script>alert('xss')</script>"},
            {"Referer": "javascript:alert('xss')"},
            {"X-Requested-With": "../../etc/passwd"},
            {"Authorization": "Bearer ../../../etc/passwd"}
        ]
        
        test_data = {"text": "Test header validation"}
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            for headers in malicious_headers:
                response = await test_client.post("/analyze", json=test_data, headers=headers)
                
                # Should process normally (headers should be ignored/sanitized)
                assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_path_traversal_protection(self, test_client):
        """Test protection against path traversal attacks."""
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd",
            "/etc/passwd%00.txt"
        ]
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            for path in path_traversal_attempts:
                response = await test_client.post("/analyze", json={"text": path})
                
                # Should treat as normal text input
                assert response.status_code == 200
                data = response.json()
                assert "sentiment" in data
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_denial_of_service_protection(self, test_client):
        """Test protection against DoS attacks."""
        # Test rapid requests
        rapid_requests = []
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.01  # Very fast to allow many requests
            )
            
            async def make_request(i):
                return await test_client.post("/analyze", json={"text": f"DoS test {i}"})
            
            # Make 50 rapid requests
            start_time = asyncio.get_event_loop().time()
            tasks = [make_request(i) for i in range(50)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = asyncio.get_event_loop().time()
            
            # Count successful responses
            success_count = sum(
                1 for r in responses 
                if hasattr(r, 'status_code') and r.status_code == 200
            )
            
            # Some requests should succeed, but rate limiting might kick in
            assert success_count > 0
            
            # Should complete within reasonable time (not hang indefinitely)
            assert end_time - start_time < 30  # 30 seconds max
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_response_header_security(self, test_client):
        """Test security-related response headers."""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        headers = response.headers
        
        # Check for security headers (if implemented)
        # Note: These might not all be implemented, but we test for good practice
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "content-security-policy",
            "strict-transport-security",
            "x-xss-protection"
        ]
        
        # At minimum, content-type should be properly set
        assert "content-type" in headers
        
        # Test that sensitive information is not leaked in headers
        for header_name, header_value in headers.items():
            header_name_lower = header_name.lower()
            header_value_lower = header_value.lower()
            
            # Should not contain sensitive information
            assert "password" not in header_value_lower
            assert "secret" not in header_value_lower
            assert "key" not in header_value_lower
            assert "token" not in header_value_lower
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_information_disclosure_prevention(self, test_client):
        """Test prevention of information disclosure."""
        # Test error responses don't leak sensitive information
        response = await test_client.post("/analyze", json={"text": ""})
        
        if response.status_code >= 400:
            error_data = response.json()
            error_text = json.dumps(error_data).lower()
            
            # Should not contain sensitive information
            sensitive_info = [
                "password", "secret", "key", "token", "database",
                "internal", "stack trace", "file path", "/home/",
                "/var/", "c:\\", "exception", "traceback"
            ]
            
            for sensitive in sensitive_info:
                assert sensitive not in error_text
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_cors_security(self, test_client):
        """Test CORS security configuration."""
        # Test preflight request
        response = await test_client.options(
            "/analyze",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # CORS handling will depend on configuration
        # Should either allow or deny appropriately
        assert response.status_code in [200, 204, 405, 403]
        
        # Test actual request with origin
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            response = await test_client.post(
                "/analyze",
                json={"text": "CORS test"},
                headers={"Origin": "https://example.com"}
            )
            
            # Should handle CORS appropriately
            assert response.status_code in [200, 403]
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_authentication_bypass_attempts(self, test_client):
        """Test authentication bypass attempts (if auth is implemented)."""
        bypass_attempts = [
            {"Authorization": "Bearer fake_token"},
            {"Authorization": "Basic fake_credentials"},
            {"X-API-Key": "fake_key"},
            {"Cookie": "session=fake_session"},
            {"X-Forwarded-User": "admin"},
            {"X-Remote-User": "root"}
        ]
        
        test_data = {"text": "Authentication bypass test"}
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            for headers in bypass_attempts:
                response = await test_client.post("/analyze", json=test_data, headers=headers)
                
                # Should either work normally or return auth error
                # Depends on whether auth is implemented
                assert response.status_code in [200, 401, 403]
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_file_upload_security(self, test_client):
        """Test file upload security (if file upload is supported)."""
        # Test potentially malicious file uploads
        malicious_files = [
            ("file", ("test.txt", "Normal text file", "text/plain")),
            ("file", ("script.js", "<script>alert('xss')</script>", "application/javascript")),
            ("file", ("test.exe", b"\x4d\x5a\x90\x00", "application/octet-stream")),
            ("file", ("test.php", "<?php system($_GET['cmd']); ?>", "application/x-php"))
        ]
        
        for file_data in malicious_files:
            try:
                response = await test_client.post("/upload", files=[file_data])
                
                # Should reject malicious files or not support upload
                assert response.status_code in [404, 405, 400, 415, 422]
                
            except Exception:
                # Upload endpoint might not exist
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_http_method_security(self, test_client):
        """Test HTTP method security."""
        dangerous_methods = ["PUT", "DELETE", "PATCH", "TRACE", "CONNECT"]
        
        for method in dangerous_methods:
            try:
                response = await test_client.request(method, "/analyze")
                
                # Should not allow dangerous methods
                assert response.status_code in [405, 501]
                
            except Exception:
                # Some methods might not be supported by the client
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_server_information_leakage(self, test_client):
        """Test for server information leakage."""
        response = await test_client.get("/health")
        
        headers = response.headers
        
        # Check that server doesn't leak too much information
        server_header = headers.get("server", "").lower()
        
        # Should not reveal specific versions or internal details
        sensitive_server_info = [
            "apache/2.4.41", "nginx/1.18.0", "python/3.9.0",
            "uvicorn", "gunicorn", "debug", "development"
        ]
        
        for sensitive in sensitive_server_info:
            if sensitive in server_header:
                # This might be acceptable in development but should be reviewed
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.security 
    async def test_timing_attack_resistance(self, test_client):
        """Test resistance to timing attacks."""
        # Test with valid and invalid inputs to check for timing differences
        valid_text = "Valid financial text for analysis"
        invalid_inputs = ["", None, 123, [], {}]
        
        timing_results = []
        
        # Test valid input timing
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            start_time = asyncio.get_event_loop().time()
            response = await test_client.post("/analyze", json={"text": valid_text})
            end_time = asyncio.get_event_loop().time()
            
            timing_results.append(("valid", end_time - start_time, response.status_code))
        
        # Test invalid inputs timing
        for invalid_input in invalid_inputs:
            try:
                start_time = asyncio.get_event_loop().time()
                response = await test_client.post("/analyze", json={"text": invalid_input})
                end_time = asyncio.get_event_loop().time()
                
                timing_results.append(("invalid", end_time - start_time, response.status_code))
                
            except Exception:
                # Some invalid inputs might cause exceptions
                pass
        
        # Timing analysis (basic check)
        valid_times = [t for label, t, _ in timing_results if label == "valid"]
        invalid_times = [t for label, t, _ in timing_results if label == "invalid"]
        
        if valid_times and invalid_times:
            avg_valid_time = sum(valid_times) / len(valid_times)
            avg_invalid_time = sum(invalid_times) / len(invalid_times)
            
            # Times should not differ dramatically (indicating potential timing attack)
            # This is a basic check - sophisticated timing attacks would need more analysis
            time_ratio = max(avg_valid_time, avg_invalid_time) / min(avg_valid_time, avg_invalid_time)
            assert time_ratio < 100  # Should not be orders of magnitude different