from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# This script pushes a metric to a Prometheus Pushgateway
# Make sure the Pushgateway is running at the specified URL

registry = CollectorRegistry()
gauge = Gauge("my_metric", "Description of my metric", registry=registry)
gauge.set(42)
push_to_gateway("http://localhost:9091", job="my_job", registry=registry)
