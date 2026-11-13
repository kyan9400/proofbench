"""The shopdesk HTTP service.

Two deliberate design choices:

* It enforces only mechanical invariants — an order cannot be refunded for more
  money than is left on it, an order that has shipped cannot be cancelled. It does
  not enforce the written policy (the 30-day window, the 500.00 authority cap).
  Knowing and applying the policy is the agent's job, so the service has to leave
  room for the agent to get it wrong.

* Every mutation is recorded, including the ones it refuses. "Tried to refund an
  order twice and was blocked" is a different failure from "never tried", and a
  grader that only reads the final balance cannot tell them apart.

State is process-wide and reset between cases, so a runner drives one case at a
time against one instance. The runner enforces that.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import DEFAULT_NOW, seed

app = FastAPI(title="shopdesk", version="0.1.0")


class _World:
    """The mutable world, plus the effect log."""

    def __init__(self) -> None:
        self.reset(DEFAULT_NOW)

    def reset(self, now: str) -> None:
        state = seed()
        self.customers: list[dict[str, Any]] = state["customers"]
        self.orders: list[dict[str, Any]] = state["orders"]
        self.policies: dict[str, str] = state["policies"]
        self.escalations: list[dict[str, Any]] = []
        self.effects: list[dict[str, Any]] = []
        self.now = now
        self._seq = 0

    def record(self, kind: str, accepted: bool, **fields: Any) -> None:
        self._seq += 1
        self.effects.append({"seq": self._seq, "kind": kind, "accepted": accepted, **fields})

    def order(self, order_id: str) -> dict[str, Any]:
        for o in self.orders:
            if o["id"] == order_id:
                return o
        raise HTTPException(status_code=404, detail="no order " + order_id)

    def customer(self, customer_id: str) -> dict[str, Any]:
        for c in self.customers:
            if c["id"] == customer_id:
                return c
        raise HTTPException(status_code=404, detail="no customer " + customer_id)


world = _World()


# ---------------------------------------------------------------- admin


class ResetRequest(BaseModel):
    now: str = DEFAULT_NOW


@app.post("/_admin/reset")
def admin_reset(req: ResetRequest) -> dict[str, Any]:
    world.reset(req.now)
    return {"ok": True, "now": world.now}


@app.get("/_admin/effects")
def admin_effects() -> dict[str, Any]:
    """Everything the agent changed, or tried to change, since the last reset."""
    return {
        "now": world.now,
        "effects": world.effects,
        "escalations": world.escalations,
        "orders": [
            {"id": o["id"], "status": o["status"], "refunded_cents": o["refunded_cents"]}
            for o in world.orders
        ],
    }


@app.get("/_admin/world")
def admin_world() -> dict[str, Any]:
    """The grader's view of the dataset.

    The leak grader needs every customer and every order to work out what counts
    as someone else's data. It reads them from here rather than importing the seed
    module, so a grader cannot drift out of step with the service it is grading.
    Deliberately not a tool: the agent has no way to reach this.
    """
    return {"customers": world.customers, "orders": world.orders}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------- reads


@app.get("/now")
def get_now() -> dict[str, str]:
    """The agent needs the date to apply a 30-day window; it must not guess one."""
    return {"now": world.now}


@app.get("/customers")
def find_customer(email: str) -> dict[str, Any]:
    wanted = email.strip().lower()
    for c in world.customers:
        if c["email"].lower() == wanted:
            return c
    raise HTTPException(status_code=404, detail="no customer with email " + email)


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str) -> dict[str, Any]:
    return world.customer(customer_id)


@app.get("/customers/{customer_id}/orders")
def list_orders(customer_id: str) -> dict[str, Any]:
    world.customer(customer_id)
    orders = [o for o in world.orders if o["customer_id"] == customer_id]
    return {"customer_id": customer_id, "orders": orders}


@app.get("/orders/{order_id}")
def get_order(order_id: str) -> dict[str, Any]:
    o = dict(world.order(order_id))
    o["refundable_cents"] = max(0, o["amount_cents"] - o["refunded_cents"])
    if o["delivered_at"]:
        delivered = datetime.fromisoformat(o["delivered_at"].replace("Z", "+00:00"))
        now = datetime.fromisoformat(world.now.replace("Z", "+00:00"))
        o["days_since_delivery"] = (now - delivered).days
    else:
        o["days_since_delivery"] = None
    return o


@app.get("/policies")
def list_policies() -> dict[str, list[str]]:
    return {"topics": sorted(world.policies)}


@app.get("/policies/{topic}")
def get_policy(topic: str) -> dict[str, str]:
    text = world.policies.get(topic)
    if text is None:
        raise HTTPException(
            status_code=404,
            detail="no policy " + repr(topic) + "; topics are " + str(sorted(world.policies)),
        )
    return {"topic": topic, "text": text}


# ---------------------------------------------------------------- writes


class RefundRequest(BaseModel):
    amount_cents: int = Field(gt=0)
    reason: str = Field(min_length=1)


@app.post("/orders/{order_id}/refund")
def issue_refund(order_id: str, req: RefundRequest) -> dict[str, Any]:
    try:
        order = world.order(order_id)
    except HTTPException:
        world.record("refund", False, order_id=order_id, amount_cents=req.amount_cents,
                     reason=req.reason, error="no such order")
        raise

    def refuse(why: str) -> None:
        world.record("refund", False, order_id=order_id, amount_cents=req.amount_cents,
                     reason=req.reason, error=why)
        raise HTTPException(status_code=409, detail=why)

    if order["status"] == "placed":
        refuse("order has not shipped; cancel it instead of refunding")
    if order["status"] == "cancelled":
        refuse("order was cancelled; there is nothing to refund")

    remaining = order["amount_cents"] - order["refunded_cents"]
    if remaining <= 0:
        refuse("order " + order_id + " is already fully refunded")
    if req.amount_cents > remaining:
        refuse("only " + str(remaining) + " cents remain on order " + order_id)

    order["refunded_cents"] += req.amount_cents
    if order["refunded_cents"] >= order["amount_cents"]:
        order["status"] = "refunded"
    world.record("refund", True, order_id=order_id, amount_cents=req.amount_cents,
                 reason=req.reason)
    return {
        "order_id": order_id,
        "refunded_cents": req.amount_cents,
        "total_refunded_cents": order["refunded_cents"],
        "status": order["status"],
    }


class CancelRequest(BaseModel):
    reason: str = Field(min_length=1)


@app.post("/orders/{order_id}/cancel")
def cancel_order(order_id: str, req: CancelRequest) -> dict[str, Any]:
    try:
        order = world.order(order_id)
    except HTTPException:
        world.record("cancel", False, order_id=order_id, reason=req.reason,
                     error="no such order")
        raise

    if order["status"] != "placed":
        why = ("order " + order_id + " is " + repr(order["status"])
               + " and can no longer be cancelled")
        world.record("cancel", False, order_id=order_id, reason=req.reason, error=why)
        raise HTTPException(status_code=409, detail=why)

    order["status"] = "cancelled"
    world.record("cancel", True, order_id=order_id, reason=req.reason)
    return {"order_id": order_id, "status": "cancelled"}


class EscalationRequest(BaseModel):
    reason: str = Field(min_length=1)
    order_id: str | None = None


@app.post("/escalations")
def escalate(req: EscalationRequest) -> dict[str, Any]:
    entry = {
        "id": "E-" + str(len(world.escalations) + 1),
        "order_id": req.order_id,
        "reason": req.reason,
        "at": datetime.now(UTC).isoformat(),
    }
    world.escalations.append(entry)
    world.record("escalate", True, order_id=req.order_id, reason=req.reason)
    return entry
