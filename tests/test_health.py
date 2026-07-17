from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_redis_status(client: TestClient) -> None:
    # Postgres isn't running in this test environment, so /ready is expected
    # to report it as down, we only care that the redis check (backed by
    # fakeredis here) comes back healthy and the endpoint doesn't crash.
    response = client.get("/ready")

    body = response.json()
    assert body["redis"] == "ok"
    assert response.status_code in (200, 503)
