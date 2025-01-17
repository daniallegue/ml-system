""" Test Consumer for the main data ingestion flow"""

from kafka import KafkaConsumer

def consume_stock_data(topic = "AAPL_stock_price", bootstrap_servers="localhost:9092"):
    """Test consumer for the AAPL stock price producer."""

    consumer =  KafkaConsumer(
        topic,
        bootstrap_servers = [bootstrap_servers],
        auto_offset_reset = 'earliest',
        group_id = 'test-consumer-group'
    )

    print(f"Listening for messages on topic '{topic}'...")
    for message in consumer:
        print(f"Received: {message.value.decode('utf-8')}")

if __name__ == "__main__":
    consume_stock_data()