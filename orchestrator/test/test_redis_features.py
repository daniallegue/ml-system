"""Simple test script for feature engineer module"""

import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

symbol = "AAPL"
redis_key = f"{symbol}_features"

features = redis_client.lrange(redis_key, 0, -1)
for feature_json in features:
    feature = json.loads(feature_json)
    print(feature)