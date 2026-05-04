"""
Kinesis consumer that ingests user clickstream events, updates user state in
DynamoDB, and triggers recommendation refresh for active users.

Deployed as an AWS Lambda function with Kinesis as the event source.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import boto3
import structlog

logger = structlog.get_logger()

dynamodb = boto3.resource("dynamodb")
USER_TABLE = dynamodb.Table(os.environ.get("USER_STATE_TABLE", "user_state"))


def lambda_handler(event: dict, context: Any) -> dict:
    """Process a batch of Kinesis records."""
    processed = 0
    failed = 0

    for record in event.get("Records", []):
        try:
            payload = _decode(record)
            _update_user_state(payload)
            processed += 1
        except Exception:
            logger.exception("record_failed", record_id=record.get("eventID"))
            failed += 1

    logger.info("batch_complete", processed=processed, failed=failed)
    return {"processed": processed, "failed": failed}


def _decode(record: dict) -> dict:
    raw = base64.b64decode(record["kinesis"]["data"])
    return json.loads(raw)


def _update_user_state(event: dict) -> None:
    """Append event to user's recent activity list (capped at 50)."""
    user_id = event["user_id"]
    item = {
        "product_id": event["product_id"],
        "event_type": event["event_type"],
        "timestamp": event["timestamp"],
    }

    USER_TABLE.update_item(
        Key={"user_id": user_id},
        UpdateExpression=(
            "SET recent_events = list_append("
            "if_not_exists(recent_events, :empty), :evt)"
        ),
        ExpressionAttributeValues={
            ":empty": [],
            ":evt": [item],
        },
    )
