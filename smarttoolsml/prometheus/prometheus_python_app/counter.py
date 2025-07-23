import http.server
from prometheus_client import start_http_server, Counter

# Define a counter metric to track the number of requests. The counter will have labels for app name and endpoint.
REQUEST_COUNTER = Counter(
    "request_count", "Total number of requests received", ["app_name", "endpoint"]
)

APP_PORT = 8000
METRICS_PORT = 8001


class HandleRequest(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Increment the request counter with labels
        REQUEST_COUNTER.labels(app_name="my_app", endpoint="/api/1").inc()
        REQUEST_COUNTER.labels(app_name="my_app", endpoint="/api/2").inc()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Hello, world!")
        self.wfile.close()


if __name__ == "__main__":
    # Prometheus will scrape metrics from this port
    # Its easy to change the port in the prometheus.yml file

    # - job_name: "python_app"
    #   static_configs:
    #       - targets: ["python-app:8001"]
    #       labels:
    #           app: "python-app"
    start_http_server(METRICS_PORT)
    server = http.server.HTTPServer(("localhost", APP_PORT), HandleRequest)
    server.serve_forever()
