""" Monitoring flow module. """

from monitor_data_shift import data_shift_monitoring_flow
from prefect import flow
from prefect.server.schemas.schedules import IntervalSchedule


import datetime


@flow
def run_monitoring_flow(symbol: str = "AAPL", window: int = 100):
    """
    Prefect flow that runs the data distribution shift monitoring.

    Parameters:
    - symbol (str): Stock symbol to monitor.
    - window (int): Number of records to use for reference data.
    """
    data_shift_monitoring_flow(symbol=symbol, window=window)


def create_and_apply_deployment():
    """
    Creates and applies a Prefect deployment for the monitoring flow.
    Schedules the flow to run every 5 minutes.
    """
    # TODO: Fix schedule interval + add to S3
    # TODO: Create default pool before deploying + agent

    run_monitoring_flow.deploy(
        name="Data Shift Monitoring Deployment",
        #schedule=IntervalSchedule(interval=datetime.timedelta(minutes=5)),
        work_pool_name="default-agent-pool",
        parameters={"symbol": "AAPL", "window": 100},
        work_queue_name="default",
        tags=["monitoring"]
    )


if __name__ == "__main__":
    # If the script is run directly, create and apply the deployment
    create_and_apply_deployment()

    #Testing purposes
    run_monitoring_flow()

