"""Structured error hierarchy for signsafe-ai workers.

Errors are split into two categories:

RetryableError
    Transient failures that are safe to retry (network blips, rate limits,
    temporary upstream unavailability).  The queue consumer will attempt up
    to MAX_RETRIES retries with exponential back-off before giving up and
    routing the message to the DLQ.

PermanentError
    Unrecoverable failures where retrying will never help (malformed message,
    missing required fields, unsupported file type, schema mismatch).  The
    queue consumer will ack the message immediately (no DLQ) after marking
    the job/analysis as failed so the problem is visible in the database.
"""

from __future__ import annotations


class SignSafeWorkerError(Exception):
    """Base class for all worker errors."""


class RetryableError(SignSafeWorkerError):
    """A transient error that may succeed on retry."""


class PermanentError(SignSafeWorkerError):
    """An unrecoverable error; retrying will not help."""
