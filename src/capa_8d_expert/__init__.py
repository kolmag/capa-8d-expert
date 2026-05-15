"""Reusable CAPA/8D Expert RAG core."""

from .answer import (
    DEFAULT_MODEL_STACK,
    MODEL_STACK_OPTIONS,
    MODEL_STACKS,
    AnswerResult,
    RankedChunk,
    RetrievedChunk,
    answer,
    answer_stream,
    resolve_model_stack,
)

__all__ = [
    "DEFAULT_MODEL_STACK",
    "MODEL_STACK_OPTIONS",
    "MODEL_STACKS",
    "AnswerResult",
    "RankedChunk",
    "RetrievedChunk",
    "answer",
    "answer_stream",
    "resolve_model_stack",
]
