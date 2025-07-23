import http.server
from prometheus_client import start_http_server, Summary

APP_PORT = 8000
METRICS_PORT = 8001

REQUEST_DURATION = Summary(
    "request_duration_seconds", "Request duration in seconds", ["app_name", "endpoint"]
)


class HandleRequest(http.server.BaseHTTPRequestHandler):

    # @REQUEST_DURATION.labels(app_name="my_app", endpoint=self.path).time() - can be used to automatically measure request duration
    def do_GET(self):
        # Measure the duration of the request
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
