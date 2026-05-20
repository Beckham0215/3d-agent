"""Measure Flask API endpoint response latency using the test client.

Tests only pure DB-backed endpoints (no Groq or BLIP calls) so the
measurement reflects database + serialisation overhead only.
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass, field
from typing import List, Optional

from eval.metrics import percentile


@dataclass
class EndpointStats:
    endpoint: str
    method: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    ok_rate: float  # fraction of 2xx responses


@dataclass
class LatencyResult:
    name: str = "latency"
    endpoints: List[EndpointStats] = field(default_factory=list)
    error: Optional[str] = None


# Number of repeated requests per endpoint
_N = 20


def _measure(client, method: str, url: str, json_body: dict) -> EndpointStats:
    latencies: List[float] = []
    ok_count = 0
    for _ in range(_N):
        t0 = time.perf_counter()
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=json_body)
        latencies.append((time.perf_counter() - t0) * 1000)
        if resp.status_code < 400:
            ok_count += 1

    return EndpointStats(
        endpoint=url,
        method=method,
        n=_N,
        p50_ms=round(percentile(latencies, 50), 2),
        p95_ms=round(percentile(latencies, 95), 2),
        p99_ms=round(percentile(latencies, 99), 2),
        mean_ms=round(sum(latencies) / _N, 2),
        min_ms=round(min(latencies), 2),
        max_ms=round(max(latencies), 2),
        ok_rate=ok_count / _N,
    )


def _seed_db(app):
    """Create a test user + space + assets so endpoints return real data."""
    from werkzeug.security import generate_password_hash
    from app.extensions import db
    from app.models import User, MatterportSpace, Asset, AssetsSummary, ScanHistory
    import json

    with app.app_context():
        db.create_all()
        user = User(
            username="eval_latency",
            email="eval_latency@test.local",
            password_hash=generate_password_hash("x"),
        )
        db.session.add(user)
        db.session.flush()

        space = MatterportSpace(
            user_id=user.user_id,
            matterport_sid="eval-sid-latency",
            map_name="Latency Test Space",
        )
        db.session.add(space)
        db.session.flush()

        for i, (label, cat) in enumerate(
            [("Kitchen 1", "kitchen"), ("Bedroom 1", "bedroom"), ("Office 1", "office")], 1
        ):
            db.session.add(
                Asset(map_id=space.map_id, label_name=label,
                      sweep_uuid=f"sweep-{i:03d}", category=cat)
            )
            db.session.add(
                AssetsSummary(map_id=space.map_id, area_name=label,
                               asset_name="chair", count=i * 2)
            )

        snapshot = json.dumps({"chair": 4, "table": 2})
        db.session.add(ScanHistory(map_id=space.map_id, area_name="Kitchen 1",
                                    snapshot=snapshot))
        db.session.commit()
        return user.user_id, space.map_id


def run() -> LatencyResult:
    # Isolate the test in a throwaway SQLite file
    tmp_db = tempfile.mktemp(suffix=".eval.db")
    orig_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db}"

    try:
        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        # Disable GROQ so no accidental API calls
        app.config["GROQ_API_KEY"] = ""

        user_id, map_id = _seed_db(app)

        stats: List[EndpointStats] = []
        with app.test_client() as client:
            # Inject an authenticated session
            with client.session_transaction() as sess:
                sess["user_id"] = user_id

            stats.append(_measure(
                client, "GET", f"/api/spaces/{map_id}/assets", {}))

            stats.append(_measure(
                client, "GET", f"/api/spaces/{map_id}/assets-panel", {}))

            stats.append(_measure(
                client, "GET",
                f"/api/spaces/{map_id}/scan-history?area=Kitchen+1", {}))

            stats.append(_measure(
                client, "GET",
                f"/api/scan-assets/locations?map_id={map_id}", {}))

        return LatencyResult(endpoints=stats)

    except Exception as exc:
        return LatencyResult(error=str(exc))

    finally:
        # Restore original DATABASE_URL
        if orig_db_url is not None:
            os.environ["DATABASE_URL"] = orig_db_url
        elif "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        try:
            os.unlink(tmp_db)
        except OSError:
            pass
