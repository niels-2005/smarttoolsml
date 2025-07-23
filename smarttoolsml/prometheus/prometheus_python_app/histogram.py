import http.server
from prometheus_client import start_http_server, Histogram

APP_PORT = 8000
METRICS_PORT = 8001


# Define a histogram metric to track request duration with labels for app name and endpoint
# buckets are defined to categorize the duration of requests
REQUEST_DURATION = Histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    ["app_name", "endpoint"],
    buckets=[0.1, 0.5, 1, 2.5, 5, 10, 30, 60],
)


class HandleRequest(http.server.BaseHTTPRequestHandler):

    # @REQUEST_DURATION.labels(app_name="my_app", endpoint=self.path).time() - can be used to automatically measure request duration
    def do_GET(self):
        with REQUEST_DURATION.labels(app_name="my_app", endpoint=self.path).time():
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
