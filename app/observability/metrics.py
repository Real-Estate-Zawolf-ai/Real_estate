"""Prometheus metrics used by API and background workflows."""

from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "real_estate_requests_total",
    "Application requests grouped by operation and outcome.",
    ("operation", "outcome"),
)

REQUEST_DURATION = Histogram(
    "real_estate_request_duration_seconds",
    "Application operation duration in seconds.",
    ("operation",),
)


def record_request(operation: str, outcome: str, duration_seconds: float) -> None:
    REQUEST_COUNT.labels(operation=operation, outcome=outcome).inc()
    REQUEST_DURATION.labels(operation=operation).observe(max(duration_seconds, 0.0))
