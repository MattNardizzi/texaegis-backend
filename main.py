from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from audit_logger import AuditLogger
from decision_engine import DecisionEngine
from deterministic_checks import DeterministicChecker
from models import EvaluationResponse, OutboundEmail
from policy import get_default_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tex_aegis")

try:
    from llm_analyzer import LLMAnalyzer, LLMAnalyzerError
except Exception:
    logger.exception("Failed to import semantic analyzer module.")
    LLMAnalyzer = None  # type: ignore[assignment]
    LLMAnalyzerError = Exception  # type: ignore[misc,assignment]


app = FastAPI(title="Tex Aegis", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

policy = get_default_policy()
deterministic_checker = DeterministicChecker()
decision_engine = DecisionEngine()
audit_logger = AuditLogger()


def _build_semantic_analyzer() -> Optional[LLMAnalyzer]:
    """
    Initialize the semantic analyzer once at startup.

    Tex must remain operational even if:
    - the semantic analyzer module is unavailable
    - its provider SDK is not installed
    - credentials are missing
    - analyzer initialization fails

    In those cases, Tex falls back to deterministic-only mode.
    """
    if LLMAnalyzer is None:
        logger.warning(
            "Semantic analyzer module unavailable. Running in deterministic-only mode."
        )
        return None

    try:
        analyzer = LLMAnalyzer()
        logger.info("Semantic analyzer initialized successfully.")
        return analyzer
    except Exception:
        logger.exception(
            "Semantic analyzer init failed. Running in deterministic-only mode."
        )
        return None


semantic_analyzer = _build_semantic_analyzer()


@app.get("/health")
def health_check() -> dict[str, str]:
    semantic_status = "enabled" if semantic_analyzer is not None else "disabled"

    return {
        "status": "ok",
        "service": "tex-aegis",
        "semantic_layer": semantic_status,
    }


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_email(email: OutboundEmail) -> EvaluationResponse:
    """
    Main Tex evaluation endpoint.

    Flow:
    1. Run deterministic checks
    2. Attempt semantic analysis if available
    3. Apply policy through the decision engine
    4. Log the result
    5. Return the full evaluation package

    Production behavior:
    - If semantic analysis fails, Tex falls back to deterministic-only mode.
    - Evaluation should still complete unless audit logging fails.
    """
    logger.info(
        "Starting evaluation for recipient=%s subject=%s",
        email.recipient,
        email.subject,
    )

    deterministic_findings = deterministic_checker.run(email=email, policy=policy)

    semantic_evaluation = None
    if semantic_analyzer is not None:
        try:
            semantic_evaluation = semantic_analyzer.analyze(email)
            logger.info("Semantic analysis completed successfully.")
        except LLMAnalyzerError as exc:
            logger.warning("Semantic analysis failed; using fallback mode: %s", exc)
            semantic_evaluation = None
        except Exception:
            logger.exception(
                "Unexpected semantic analysis failure; using fallback mode."
            )
            semantic_evaluation = None
    else:
        logger.info(
            "Semantic analyzer unavailable for this request. Using deterministic-only mode."
        )

    final_decision = decision_engine.decide(
        deterministic_findings=deterministic_findings,
        semantic_evaluation=semantic_evaluation,
        policy=policy,
    )

    evaluation_response = EvaluationResponse(
        email=email,
        deterministic_findings=deterministic_findings,
        semantic_evaluation=semantic_evaluation,
        final_decision=final_decision,
    )

    try:
        audit_logger.log_evaluation(
            evaluation_response,
            policy_version=policy.version,
        )
    except Exception as exc:
        logger.exception("Audit logging failed after evaluation completed.")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation completed but audit logging failed: {exc}",
        ) from exc

    logger.info(
        "Evaluation completed. decision=%s semantic_enabled=%s semantic_returned=%s findings=%s",
        final_decision.decision,
        semantic_analyzer is not None,
        semantic_evaluation is not None,
        len(deterministic_findings),
    )

    return evaluation_response