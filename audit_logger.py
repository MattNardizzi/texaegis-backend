from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from models import EvaluationResponse


class AuditLogger:
    """
    Append-only audit logger for Tex Aegis evaluations.

    Each evaluation is written as one JSON object per line (JSONL).
    Every record includes a deterministic decision_id derived from
    the email content + policy version, making each decision
    referenceable and replayable.
    """

    def __init__(self, log_path: str = "decision_log.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_evaluation(
        self, evaluation: EvaluationResponse, policy_version: str
    ) -> str:
        """
        Append one evaluation record to the audit log.
        Returns the decision_id.
        """
        record = self._build_record(evaluation, policy_version)

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record["decision_id"]

    def _build_record(
        self, evaluation: EvaluationResponse, policy_version: str
    ) -> Dict[str, Any]:
        """
        Build one normalized audit record with a deterministic decision_id.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        decision_id = self._generate_decision_id(
            evaluation=evaluation,
            policy_version=policy_version,
            timestamp=timestamp,
        )

        return {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "policy_version": policy_version,
            "email": evaluation.email.model_dump(mode="json"),
            "deterministic_findings": [
                finding.model_dump(mode="json")
                for finding in evaluation.deterministic_findings
            ],
            "semantic_evaluation": (
                evaluation.semantic_evaluation.model_dump(mode="json")
                if evaluation.semantic_evaluation is not None
                else None
            ),
            "final_decision": evaluation.final_decision.model_dump(mode="json"),
        }

    @staticmethod
    def _generate_decision_id(
        evaluation: EvaluationResponse,
        policy_version: str,
        timestamp: str,
    ) -> str:
        """
        Deterministic decision ID from email content + policy version + timestamp.
        Same inputs always produce the same ID.
        """
        payload = json.dumps(
            {
                "email": evaluation.email.model_dump(mode="json"),
                "policy_version": policy_version,
                "timestamp": timestamp,
            },
            sort_keys=True,
        )
        hash_hex = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"tex_{hash_hex[:16]}"