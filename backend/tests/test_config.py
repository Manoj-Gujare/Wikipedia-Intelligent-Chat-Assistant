"""Configuration guards and retry classification."""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

from app.config import Settings
from app.core.embeddings import is_retryable


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def test_missing_key_is_not_configured():
    assert _settings(openai_api_key="").openai_configured is False


def test_placeholder_key_is_not_configured():
    # The exact string shipped in .env.example — copying it without editing is
    # the most common first-run mistake.
    assert _settings(openai_api_key="sk-your-key-here").openai_configured is False


def test_real_looking_key_is_configured():
    assert _settings(openai_api_key="sk-proj-abc123").openai_configured is True


def test_cors_origins_are_split_and_trimmed():
    settings = _settings(
        openai_api_key="sk-test",
        cors_origins="http://localhost:3000, https://example.com ",
    )

    assert settings.cors_origin_list == ["http://localhost:3000", "https://example.com"]


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(code, request=request)
    if code == 429:
        return RateLimitError("rate limited", response=response, body=None)
    return APIStatusError("boom", response=response, body=None)


@pytest.mark.parametrize("code", [500, 502, 503])
def test_server_errors_are_retryable(code):
    assert is_retryable(_status_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retryable(code):
    # Retrying a bad key just delays the error the operator needs to see.
    assert is_retryable(_status_error(code)) is False


def test_rate_limits_are_retryable():
    assert is_retryable(_status_error(429)) is True


def test_connection_errors_are_retryable():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")

    assert is_retryable(APIConnectionError(request=request)) is True


def test_unrelated_exceptions_are_not_retryable():
    assert is_retryable(ValueError("nope")) is False
