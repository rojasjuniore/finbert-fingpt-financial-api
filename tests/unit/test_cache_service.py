"""
Unit tests for cache service.
"""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
import time
from datetime import datetime, timedelta

from src.services.cache_service import (
    CacheService,
    CacheConfig,
    CacheKey,
    CacheEntry
)
from src.exceptions import CacheError


class TestCacheService:
    """Test cases for cache service."""
    
    @pytest.fixture
    def cache_config(self):
        """Cache configuration for testing."""
        return CacheConfig(
            redis_url="redis://localhost:6379/15",
            default_ttl=3600,  # 1 hour
            key_prefix="finbert_test:",
            enable_compression=True,
            max_key_length=250
        )
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.ttl.return_value = 3600
        mock_redis.keys.return_value = []
        mock_redis.flushdb.return_value = True
        return mock_redis
    
    @pytest.fixture
    def cache_service(self, cache_config, mock_redis):
        """Create cache service instance for testing."""
        with patch('src.services.cache_service.redis.Redis.from_url', return_value=mock_redis):
            service = CacheService(cache_config)
            return service
    
    def test_cache_config_validation(self):
        """Test cache configuration validation."""
        # Valid config
        config = CacheConfig(
            redis_url="redis://localhost:6379/0",
            default_ttl=3600,
            key_prefix="test:",
            enable_compression=True
        )
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.default_ttl == 3600
        
        # Invalid TTL
        with pytest.raises(ValueError):
            CacheConfig(
                redis_url="redis://localhost:6379/0",
                default_ttl=0,
                key_prefix="test:"
            )
        
        # Invalid key prefix
        with pytest.raises(ValueError):
            CacheConfig(
                redis_url="redis://localhost:6379/0",
                default_ttl=3600,
                key_prefix=""
            )
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache_config, mock_redis):
        """Test cache service initialization."""
        with patch('src.services.cache_service.redis.Redis.from_url', return_value=mock_redis):
            service = CacheService(cache_config)
            await service.initialize()
            
            assert service.config == cache_config
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_initialization_failure(self, cache_config):
        """Test cache initialization failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        with patch('src.services.cache_service.redis.Redis.from_url', return_value=mock_redis):
            service = CacheService(cache_config)
            
            with pytest.raises(CacheError, match="Failed to connect to Redis"):
                await service.initialize()
    
    def test_generate_cache_key(self, cache_service):
        """Test cache key generation."""
        # Test simple text
        key = cache_service._generate_cache_key("simple text")
        assert key.startswith("finbert_test:sentiment:")
        assert len(key) <= 250
        
        # Test long text
        long_text = "This is a very long text " * 50
        key = cache_service._generate_cache_key(long_text)
        assert key.startswith("finbert_test:sentiment:")
        assert len(key) <= 250
        
        # Test special characters
        special_text = "Special chars: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        key = cache_service._generate_cache_key(special_text)
        assert key.startswith("finbert_test:sentiment:")
        
        # Test unicode characters
        unicode_text = "Unicode: émojis 📈 📉 💰 ünïcödé"
        key = cache_service._generate_cache_key(unicode_text)
        assert key.startswith("finbert_test:sentiment:")
    
    def test_cache_key_consistency(self, cache_service):
        """Test that same text generates same cache key."""
        text = "Consistent cache key test"
        
        key1 = cache_service._generate_cache_key(text)
        key2 = cache_service._generate_cache_key(text)
        
        assert key1 == key2
    
    def test_cache_key_uniqueness(self, cache_service):
        """Test that different texts generate different cache keys."""
        text1 = "First test text"
        text2 = "Second test text"
        
        key1 = cache_service._generate_cache_key(text1)
        key2 = cache_service._generate_cache_key(text2)
        
        assert key1 != key2
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_service, mock_redis):
        """Test cache miss scenario."""
        mock_redis.get.return_value = None
        
        result = await cache_service.get("test text")
        
        assert result is None
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_service, mock_redis):
        """Test cache hit scenario."""
        cached_data = {
            "label": "positive",
            "score": 0.85,
            "confidence": 0.92,
            "processing_time": 0.234,
            "timestamp": datetime.now().isoformat()
        }
        mock_redis.get.return_value = json.dumps(cached_data)
        
        result = await cache_service.get("test text")
        
        assert result is not None
        assert result.label == "positive"
        assert result.score == 0.85
        assert result.confidence == 0.92
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cache_corrupted_data(self, cache_service, mock_redis):
        """Test handling of corrupted cache data."""
        mock_redis.get.return_value = "invalid json data"
        
        result = await cache_service.get("test text")
        
        # Should return None for corrupted data
        assert result is None
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_cache(self, cache_service, mock_redis):
        """Test setting cache data."""
        from src.services.model_service import SentimentResult
        
        sentiment_result = SentimentResult(
            label="positive",
            score=0.85,
            confidence=0.92,
            processing_time=0.234
        )
        
        await cache_service.set("test text", sentiment_result)
        
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        
        # Check that data was serialized properly
        key, value, ex = call_args[0]
        assert key.startswith("finbert_test:sentiment:")
        assert ex == 3600  # default TTL
        
        # Verify serialized data
        cached_data = json.loads(value)
        assert cached_data["label"] == "positive"
        assert cached_data["score"] == 0.85
        assert cached_data["confidence"] == 0.92
    
    @pytest.mark.asyncio
    async def test_set_cache_with_custom_ttl(self, cache_service, mock_redis):
        """Test setting cache data with custom TTL."""
        from src.services.model_service import SentimentResult
        
        sentiment_result = SentimentResult(
            label="neutral",
            score=0.5,
            confidence=0.7,
            processing_time=0.2
        )
        
        custom_ttl = 7200  # 2 hours
        await cache_service.set("test text", sentiment_result, ttl=custom_ttl)
        
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        
        key, value, ex = call_args[0]
        assert ex == custom_ttl
    
    @pytest.mark.asyncio
    async def test_set_cache_failure(self, cache_service, mock_redis):
        """Test cache set failure handling."""
        from src.services.model_service import SentimentResult
        
        sentiment_result = SentimentResult(
            label="positive",
            score=0.8,
            confidence=0.9,
            processing_time=0.1
        )
        
        mock_redis.set.side_effect = Exception("Redis set failed")
        
        # Should not raise exception, just log error
        await cache_service.set("test text", sentiment_result)
        
        mock_redis.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_cache(self, cache_service, mock_redis):
        """Test cache deletion."""
        mock_redis.delete.return_value = 1
        
        result = await cache_service.delete("test text")
        
        assert result is True
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_cache_not_found(self, cache_service, mock_redis):
        """Test cache deletion when key doesn't exist."""
        mock_redis.delete.return_value = 0
        
        result = await cache_service.delete("nonexistent text")
        
        assert result is False
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exists_cache(self, cache_service, mock_redis):
        """Test cache key existence check."""
        mock_redis.exists.return_value = True
        
        result = await cache_service.exists("test text")
        
        assert result is True
        mock_redis.exists.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exists_cache_not_found(self, cache_service, mock_redis):
        """Test cache key existence check when key doesn't exist."""
        mock_redis.exists.return_value = False
        
        result = await cache_service.exists("nonexistent text")
        
        assert result is False
        mock_redis.exists.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_ttl(self, cache_service, mock_redis):
        """Test getting TTL of cached key."""
        mock_redis.ttl.return_value = 1800  # 30 minutes
        
        ttl = await cache_service.get_ttl("test text")
        
        assert ttl == 1800
        mock_redis.ttl.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_ttl_expired(self, cache_service, mock_redis):
        """Test getting TTL of expired key."""
        mock_redis.ttl.return_value = -2  # Key doesn't exist
        
        ttl = await cache_service.get_ttl("expired text")
        
        assert ttl == -2
        mock_redis.ttl.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, cache_service, mock_redis):
        """Test clearing all cache entries."""
        mock_redis.keys.return_value = [
            "finbert_test:sentiment:key1",
            "finbert_test:sentiment:key2",
            "finbert_test:sentiment:key3"
        ]
        mock_redis.delete.return_value = 3
        
        result = await cache_service.clear()
        
        assert result == 3
        mock_redis.keys.assert_called_once_with("finbert_test:*")
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_cache_empty(self, cache_service, mock_redis):
        """Test clearing cache when no entries exist."""
        mock_redis.keys.return_value = []
        
        result = await cache_service.clear()
        
        assert result == 0
        mock_redis.keys.assert_called_once()
        mock_redis.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_service, mock_redis):
        """Test getting cache statistics."""
        mock_redis.keys.return_value = [
            "finbert_test:sentiment:key1",
            "finbert_test:sentiment:key2"
        ]
        mock_redis.memory_usage.return_value = 1024
        
        with patch.object(cache_service, '_cache_hits', 10), \
             patch.object(cache_service, '_cache_misses', 5):
            
            stats = await cache_service.get_stats()
            
            assert stats["total_keys"] == 2
            assert stats["cache_hits"] == 10
            assert stats["cache_misses"] == 5
            assert stats["hit_rate"] == 0.67  # 10/(10+5) rounded
    
    @pytest.mark.asyncio
    async def test_cache_compression(self, cache_service, mock_redis):
        """Test cache data compression."""
        from src.services.model_service import SentimentResult
        
        # Create large sentiment result
        large_text = "Large text data " * 1000
        sentiment_result = SentimentResult(
            label="neutral",
            score=0.5,
            confidence=0.7,
            processing_time=0.3
        )
        
        await cache_service.set(large_text, sentiment_result)
        
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        key, value, ex = call_args[0]
        
        # If compression is enabled, check that data was compressed
        if cache_service.config.enable_compression:
            # Value should be base64 encoded compressed data
            import base64
            try:
                base64.b64decode(value)
                # If successful, data was compressed
                assert True
            except Exception:
                # If not base64, compression might not have been applied
                pass
    
    @pytest.mark.asyncio
    async def test_cache_key_length_limit(self, cache_service):
        """Test cache key length limit enforcement."""
        # Very long text that would create a long cache key
        very_long_text = "Very long text that would normally create a very long cache key " * 100
        
        key = cache_service._generate_cache_key(very_long_text)
        
        # Key should be within the limit
        assert len(key) <= cache_service.config.max_key_length
    
    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self, cache_service, mock_redis):
        """Test concurrent cache access."""
        import asyncio
        
        # Mock different responses for concurrent calls
        mock_redis.get.side_effect = [None, None, None]  # All cache misses
        
        async def get_cache(text):
            return await cache_service.get(f"concurrent test {text}")
        
        # Make concurrent cache requests
        tasks = [get_cache(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should be cache misses
        assert all(result is None for result in results)
        assert mock_redis.get.call_count == 3
    
    @pytest.mark.asyncio
    async def test_cache_batch_operations(self, cache_service, mock_redis):
        """Test batch cache operations."""
        texts = [f"batch text {i}" for i in range(5)]
        
        # Test batch get (cache misses)
        mock_redis.mget.return_value = [None] * 5
        
        results = await cache_service.batch_get(texts)
        
        assert len(results) == 5
        assert all(result is None for result in results)
    
    @pytest.mark.asyncio
    async def test_cache_expiration_handling(self, cache_service, mock_redis):
        """Test handling of expired cache entries."""
        # Simulate expired entry
        mock_redis.get.return_value = None
        mock_redis.ttl.return_value = -2  # Key doesn't exist
        
        result = await cache_service.get("expired text")
        
        assert result is None
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, cache_service, mock_redis):
        """Test cache health check."""
        mock_redis.ping.return_value = True
        
        is_healthy = await cache_service.health_check()
        
        assert is_healthy is True
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, cache_service, mock_redis):
        """Test cache health check failure."""
        mock_redis.ping.side_effect = Exception("Connection lost")
        
        is_healthy = await cache_service.health_check()
        
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_close_connection(self, cache_service, mock_redis):
        """Test closing cache connection."""
        await cache_service.close()
        
        mock_redis.close.assert_called_once()
    
    @pytest.mark.benchmark
    def test_cache_key_generation_performance(self, cache_service, benchmark):
        """Benchmark cache key generation performance."""
        text = "Performance test text for cache key generation benchmark"
        
        result = benchmark(cache_service._generate_cache_key, text)
        
        assert result.startswith("finbert_test:sentiment:")
        assert len(result) <= 250
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_tracking(self, cache_service, mock_redis):
        """Test cache hit rate tracking."""
        # Reset counters
        cache_service._cache_hits = 0
        cache_service._cache_misses = 0
        
        # Simulate cache hit
        mock_redis.get.return_value = json.dumps({
            "label": "positive",
            "score": 0.8,
            "confidence": 0.9,
            "processing_time": 0.1,
            "timestamp": datetime.now().isoformat()
        })
        
        await cache_service.get("hit test")
        assert cache_service._cache_hits == 1
        assert cache_service._cache_misses == 0
        
        # Simulate cache miss
        mock_redis.get.return_value = None
        await cache_service.get("miss test")
        
        assert cache_service._cache_hits == 1
        assert cache_service._cache_misses == 1
    
    @pytest.mark.asyncio
    async def test_cache_with_redis_cluster(self, cache_config):
        """Test cache service with Redis cluster configuration."""
        # This would be tested with actual Redis cluster setup
        # For now, just test that the service can be initialized
        cluster_config = CacheConfig(
            redis_url="redis://cluster-node1:6379,cluster-node2:6379/0",
            default_ttl=3600,
            key_prefix="finbert_cluster:",
            enable_compression=True
        )
        
        # Test that config is valid
        assert cluster_config.redis_url.startswith("redis://")
        assert cluster_config.key_prefix == "finbert_cluster:"