"""Data Ingestion Flow tasks using Prefect and Kafka"""

import os
import time
import requests
from kafka import KafkaProducer
from prefect import flow, task

@task
def fetch_stock_data(symbol : str, api_key : str) -> dict:
    """
    Fetch stock data from AlphaVantage API

    :return: JSON response if successful
    """
    url = ( # Fetch intra-day values for a specific stock
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}"
        f"&interval=1min"
        f"&apikey={api_key}"
    )

    res = requests.get(url)
    res.raise_for_status() # Might raise error
    data = res.json()

    return data

@task
def send_data_to_kafka(data : dict, topic : str, bootstrap_servers : str = "localhost:9092"):
    """
    Sends data to a specified Kafka topic
    """

    producer = KafkaProducer(
        bootstrap_servers = [bootstrap_servers],
        value_serializer = lambda v : str(v).encode('utf-8')
    )

    producer.send(topic, value = data)
    producer.flush()
    producer.close()


@flow
def ingest_data_full_flow():
    """
    Define the flow that fetches stock data and sends it to the Kafka queue
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "DEFAULT_KEY")
    print(api_key)

    #if api_key == "DEFAULT_KEY":
    # TODO: error handling

    symbol = "AAPL"
    topic = f"{symbol}_stock_price"

    stock_data = fetch_stock_data(symbol, topic)

    send_data_to_kafka(stock_data, topic)

    print(f"Data for {symbol} sent to Kafka topic '{topic}'.")


if __name__ == "__main__":
    while True:
        # Fetch data every minute
        ingest_data_full_flow()
        time.sleep(60)


