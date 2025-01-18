""" Simple script to check Redis records """


import os
import redis
from dotenv import load_dotenv

def clear_feature_stores(symbol_pattern="*_features"):
    """
    Deletes all Redis keys that match the given pattern.

    Parameters:
    - symbol_pattern (str): The pattern to match feature store keys.
                             Default is "*_features".
    """
    load_dotenv()

    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        # Test connection
        redis_client.ping()
        print("Connected to Redis successfully.")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}")
        return

    # Use SCAN to iterate through keys safely
    cursor = 0
    keys_to_delete = []
    while True:
        cursor, keys = redis_client.scan(cursor=cursor, match=symbol_pattern, count=1000)
        keys_to_delete.extend(keys)
        if cursor == 0:
            break

    if not keys_to_delete:
        print("No feature store keys found to delete.")
        return

    # Delete the keys
    deleted = redis_client.delete(*keys_to_delete)
    print(f"Deleted {deleted} feature store keys.")

if __name__ == "__main__":
    clear_feature_stores()