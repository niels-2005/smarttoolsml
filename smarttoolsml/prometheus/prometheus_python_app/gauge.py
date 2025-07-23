import http.server
from prometheus_client import start_http_server, Gauge


REQUEST_IN_PROGRESS = Gauge(
    "request_in_progress", "Number of requests in progress", ["app_name", "endpoint"]
)

APP_PORT = 8000
METRICS_PORT = 8001


class HandleRequest(http.server.BaseHTTPRequestHandler):

    # @REQUEST_IN_PROGRESS.track_inprogress() - can be used to automatically track in-progress requests
    def do_GET(self):
        # Increment the request in progress gauge with labels
        REQUEST_IN_PROGRESS.labels(app_name="my_app", endpoint=self.path).inc()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Hello, world!")
        self.wfile.close()
        # Decrement the request in progress gauge after handling the request
        REQUEST_IN_PROGRESS.labels(app_name="my_app", endpoint=self.path).dec()

        # Reset to zero after request completion
        REQUEST_IN_PROGRESS.labels(app_name="my_app", endpoint=self.path).set(0)


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
