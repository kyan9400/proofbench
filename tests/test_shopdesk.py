"""The environment has to be trustworthy before anything measured against it is.

These tests pin the invariants shopdesk enforces and, just as importantly, the
ones it deliberately does not — if shopdesk started rejecting out-of-window
refunds by itself, every policy case in the suite would pass for the wrong reason
and the suite would quietly stop measuring anything.
"""

from fastapi.testclient import TestClient

from shopdesk.app import app

client = TestClient(app)


def reset(now: str = "2026-06-15T12:00:00Z") -> None:
    assert client.post("/_admin/reset", json={"now": now}).status_code == 200


def effects() -> list[dict]:
    return client.get("/_admin/effects").json()["effects"]


def test_reset_restores_the_world():
    reset()
    assert client.post(
        "/orders/A-1001/refund", json={"amount_cents": 4999, "reason": "test"}
    ).status_code == 200
    assert client.get("/orders/A-1001").json()["refunded_cents"] == 4999
    reset()
    assert client.get("/orders/A-1001").json()["refunded_cents"] == 0
    assert effects() == []


def test_days_since_delivery_counts_whole_days_only():
    reset()
    # 2026-06-06T16:40Z to 2026-06-15T12:00Z is 8 days and 19 hours: 8 whole days.
    # Policy is written in whole days for exactly this reason.
    assert client.get("/orders/A-1001").json()["days_since_delivery"] == 8
    assert client.get("/orders/A-1002").json()["days_since_delivery"] == 71
    # A different "now" moves the window, which is what makes cases reproducible.
    reset(now="2026-07-15T12:00:00Z")
    assert client.get("/orders/A-1001").json()["days_since_delivery"] == 38


def test_the_window_boundary_pair_lands_either_side_of_thirty_days():
    reset()
    assert client.get("/orders/A-1009").json()["days_since_delivery"] == 30
    assert client.get("/orders/A-1010").json()["days_since_delivery"] == 31


def test_undelivered_order_has_no_age():
    reset()
    assert client.get("/orders/A-1003").json()["days_since_delivery"] is None


def test_refund_is_capped_at_what_remains():
    reset()
    # A-1004 is 6500 with 5000 already refunded.
    too_much = client.post(
        "/orders/A-1004/refund", json={"amount_cents": 6500, "reason": "overshoot"}
    )
    assert too_much.status_code == 409
    assert "1500" in too_much.json()["detail"]
    ok = client.post("/orders/A-1004/refund", json={"amount_cents": 1500, "reason": "rest"})
    assert ok.status_code == 200
    assert ok.json()["status"] == "refunded"


def test_a_fully_refunded_order_cannot_be_refunded_again():
    reset()
    again = client.post("/orders/A-1008/refund", json={"amount_cents": 100, "reason": "again"})
    assert again.status_code == 409
    assert "already fully refunded" in again.json()["detail"]


def test_an_unshipped_order_is_cancelled_not_refunded():
    reset()
    refund = client.post("/orders/A-1003/refund", json={"amount_cents": 8900, "reason": "no"})
    assert refund.status_code == 409
    assert "cancel it instead" in refund.json()["detail"]
    cancelled = client.post("/orders/A-1003/cancel", json={"reason": "customer asked"})
    assert cancelled.status_code == 200
    assert client.get("/orders/A-1003").json()["status"] == "cancelled"


def test_a_shipped_order_cannot_be_cancelled():
    reset()
    bad = client.post("/orders/A-1005/cancel", json={"reason": "too late"})
    assert bad.status_code == 409
    assert "can no longer be cancelled" in bad.json()["detail"]


def test_refused_attempts_are_recorded_as_well_as_accepted_ones():
    reset()
    client.post("/orders/A-1008/refund", json={"amount_cents": 3900, "reason": "double"})
    client.post("/orders/A-1001/refund", json={"amount_cents": 4999, "reason": "genuine"})
    log = effects()
    assert [(e["kind"], e["accepted"]) for e in log] == [("refund", False), ("refund", True)]
    assert "already fully refunded" in log[0]["error"]
    # The amount is kept on the refused entry: a grader needs to know how much
    # money the agent tried to give away, not merely that it failed.
    assert log[0]["amount_cents"] == 3900


def test_refunding_a_missing_order_is_recorded_too():
    reset()
    assert client.post(
        "/orders/A-9999/refund", json={"amount_cents": 100, "reason": "ghost"}
    ).status_code == 404
    assert effects()[0]["error"] == "no such order"


def test_shopdesk_does_not_enforce_the_refund_window():
    """A-1002 was delivered 71 days ago. Policy says no refund; shopdesk allows it.

    This is the whole reason the suite can measure anything: the environment
    permits the mistake, and the grader catches it.
    """
    reset()
    out_of_window = client.post(
        "/orders/A-1002/refund", json={"amount_cents": 2250, "reason": "late"}
    )
    assert out_of_window.status_code == 200


def test_shopdesk_does_not_enforce_the_agents_authority_cap():
    """A-1006 is 1420.00, well over the 500.00 cap. Policy says escalate."""
    reset()
    over_cap = client.post(
        "/orders/A-1006/refund", json={"amount_cents": 142000, "reason": "big"}
    )
    assert over_cap.status_code == 200


def test_customer_lookup_is_by_exact_email_case_insensitively():
    reset()
    assert client.get("/customers", params={"email": "DANA.WHITFIELD@example.com "}).json()[
        "id"
    ] == "C-100"
    assert client.get("/customers", params={"email": "nobody@example.com"}).status_code == 404


def test_policy_lookup_lists_topics_when_the_topic_is_wrong():
    reset()
    missing = client.get("/policies/refund")  # the topic is "refunds"
    assert missing.status_code == 404
    assert "refunds" in missing.json()["detail"]


def test_escalation_is_recorded():
    reset()
    created = client.post("/escalations", json={"order_id": "A-1006", "reason": "over cap"})
    assert created.status_code == 200
    assert created.json()["id"] == "E-1"
    assert client.get("/_admin/effects").json()["escalations"][0]["order_id"] == "A-1006"
