"""Multi-model provider integration and fallback coordination for Codebase Analyzer.

Supports Groq as the primary provider with automatic multi-model fallback,
and Google Gemini as an optional final backup.
"""

import os
import re
from typing import Any, Callable, List, Optional, Tuple

from dotenv import load_dotenv

from src.models import SearchResult

StatusCallback = Callable[[str], None]


def _report_status(callback: Optional[StatusCallback], message: str) -> None:
    """Invokes the status callback safely if provided."""
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


def _sanitize_error_message(err: Exception) -> str:
    """Removes sensitive credentials or tokens from error messages."""
    msg = str(err)
    return re.sub(r"(api_?[kK]ey|[kK]ey|token|auth)=[\w.-]+", r"\1=***", msg)


def get_groq_api_key(api_key: Optional[str] = None) -> str:
    """Retrieves Groq API key with support for GROQ_API_KEY and groq_api_keys."""
    load_dotenv()
    if api_key is not None:
        return api_key.strip()
    return (
        os.getenv("GROQ_API_KEY")
        or os.getenv("groq_api_keys")
        or os.getenv("GROQ_API_KEYS")
        or ""
    ).strip()


def get_gemini_api_key(api_key: Optional[str] = None) -> str:
    """Retrieves Gemini API key from environment."""
    load_dotenv()
    if api_key is not None:
        return api_key.strip()
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def get_groq_client(api_key: Optional[str] = None, timeout: float = 30.0) -> Any:
    """Initializes and returns an official Groq client."""
    from groq import Groq
    from src.llm import MissingAPIKeyError

    key = get_groq_api_key(api_key)
    if not key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY (or groq_api_keys) is missing or empty. "
            "Please configure GROQ_API_KEY in your .env file."
        )
    return Groq(api_key=key, timeout=timeout)


def get_gemini_client(api_key: Optional[str] = None) -> Any:
    """Initializes and returns an official Google Gemini client."""
    from google import genai
    from src.llm import MissingAPIKeyError

    key = get_gemini_api_key(api_key)
    if not key:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY environment variable is missing or empty. "
            "Please configure GEMINI_API_KEY in your environment or local .env file."
        )
    return genai.Client(api_key=key)


def classify_groq_error(err: Exception) -> Tuple[bool, str]:
    """Classifies a Groq exception as retryable/fallback-eligible or fatal.

    Returns:
        (is_retryable, short_reason)
    """
    from groq import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
    )
    from src.llm import EmptyResponseError

    if isinstance(err, EmptyResponseError):
        return True, "Empty response from model"
    if isinstance(err, AuthenticationError):
        return False, "Authentication failed (invalid Groq API key)"
    if isinstance(err, PermissionDeniedError):
        return False, "Permission denied (Groq API key)"
    if isinstance(err, RateLimitError):
        return True, "Rate limit reached (HTTP 429)"
    if isinstance(err, InternalServerError):
        return True, "Server error (HTTP 5xx)"
    if isinstance(err, NotFoundError):
        return True, "Model not found or unavailable (HTTP 404)"
    if isinstance(err, APITimeoutError):
        return True, "Request timed out"
    if isinstance(err, APIConnectionError):
        return True, "Network connection error"
    if isinstance(err, APIStatusError):
        code = getattr(err, "status_code", None)
        if code in (401, 403):
            return False, f"Auth/Permission error (HTTP {code})"
        if code in (404, 429, 500, 502, 503, 504):
            return True, f"HTTP {code} error"
        return False, f"APIStatusError HTTP {code}"
    if isinstance(err, BadRequestError):
        msg = str(err).lower()
        if any(term in msg for term in ("context_length_exceeded", "too large", "prompt is too long")):
            return True, "Context length exceeded"
        return False, "Malformed request (HTTP 400)"

    return False, type(err).__name__


def call_groq_model(
    client: Any,
    model: str,
    prompt: str,
    system_instruction: str,
    max_output_tokens: int = 2048,
    temperature: float = 0.2,
    timeout: float = 30.0,
) -> str:
    """Calls a single Groq model and returns its response content."""
    from src.llm import EmptyResponseError

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
        timeout=timeout,
    )

    if not response.choices or not response.choices[0].message:
        raise EmptyResponseError(f"Groq model {model} returned an empty response.")

    content = getattr(response.choices[0].message, "content", "") or ""
    if not content.strip():
        raise EmptyResponseError(f"Groq model {model} returned whitespace content.")

    return content.strip()


def execute_groq_attempt(
    client: Any,
    model: str,
    query: str,
    results: List[SearchResult],
    system_instruction: str,
    max_output_tokens: int = 2048,
    temperature: float = 0.2,
    timeout: float = 30.0,
) -> str:
    """Attempts to generate an answer with a Groq model, handling context reduction if needed."""
    from src.llm import build_rag_prompt

    prompt = build_rag_prompt(query=query, results=results)
    try:
        return call_groq_model(
            client=client,
            model=model,
            prompt=prompt,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    except Exception as err:
        is_retryable, reason = classify_groq_error(err)
        if is_retryable and "Context length" in reason:
            # Safely reduce context to half and retry once on this model
            reduced_prompt = build_rag_prompt(query=query, results=results, max_context_chars=8000)
            return call_groq_model(
                client=client,
                model=model,
                prompt=reduced_prompt,
                system_instruction=system_instruction,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        raise err


def call_gemini_model(
    client: Any,
    model: str,
    prompt: str,
    system_instruction: str,
    max_output_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """Calls Google Gemini model and returns its response content."""
    from google.genai import types
    from google.genai.errors import APIError
    from src.llm import EmptyResponseError, GeminiAPIError

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except APIError as err:
        cleaned = _sanitize_error_message(err)
        raise GeminiAPIError(f"Gemini API error ({getattr(err, 'code', None) or 'unknown'}): {cleaned}") from err
    except Exception as err:
        cleaned = _sanitize_error_message(err)
        raise GeminiAPIError(f"Failed to communicate with Gemini: {cleaned}") from err

    raw_text = getattr(response, "text", "") or ""
    if not raw_text.strip():
        raise EmptyResponseError(f"Gemini model {model} returned an empty response.")

    return raw_text.strip()


def execute_multi_provider_fallback(
    query: str,
    results: List[SearchResult],
    groq_client: Optional[Any] = None,
    gemini_client: Optional[Any] = None,
    groq_primary_model: Optional[str] = None,
    groq_fallback_models: Optional[List[str]] = None,
    enable_gemini_fallback: Optional[bool] = None,
    gemini_model: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    system_instruction: str = "",
    max_output_tokens: int = 2048,
    temperature: float = 0.2,
    timeout: float = 30.0,
    on_status: Optional[StatusCallback] = None,
) -> "RAGResponse":
    """Orchestrates multi-model execution starting with Groq and falling back gracefully."""
    from src.llm import (
        DEFAULT_ENABLE_GEMINI_FALLBACK,
        DEFAULT_GEMINI_MODEL,
        DEFAULT_GROQ_FALLBACK_MODELS,
        DEFAULT_GROQ_PRIMARY_MODEL,
        AllProvidersFailedError,
        MissingAPIKeyError,
        RAGResponse,
        build_rag_prompt,
    )

    primary_model = (
        groq_primary_model or os.getenv("GROQ_PRIMARY_MODEL", DEFAULT_GROQ_PRIMARY_MODEL)
    ).strip()

    if groq_fallback_models is not None:
        configured_fallbacks = list(groq_fallback_models)
    else:
        env_fallbacks = os.getenv("GROQ_FALLBACK_MODELS", ",".join(DEFAULT_GROQ_FALLBACK_MODELS))
        configured_fallbacks = [m.strip() for m in env_fallbacks.split(",") if m.strip()]

    # Deduplicate fallback models and exclude primary model
    groq_model_sequence = [primary_model]
    for m in configured_fallbacks:
        if m not in groq_model_sequence:
            groq_model_sequence.append(m)

    # Determine Gemini fallback availability
    if enable_gemini_fallback is not None:
        gemini_allowed = bool(enable_gemini_fallback)
    else:
        env_gem_fallback = os.getenv("ENABLE_GEMINI_FALLBACK", str(DEFAULT_ENABLE_GEMINI_FALLBACK))
        gemini_allowed = env_gem_fallback.lower() in ("true", "1", "yes")

    target_gemini_model = (
        gemini_model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    ).strip()

    fallback_trace: List[str] = []
    groq_key = get_groq_api_key(groq_api_key)

    # Check for missing Groq credentials
    if groq_client is None and not groq_key:
        if gemini_allowed:
            gem_key = get_gemini_api_key(gemini_api_key)
            if not gem_key and gemini_client is None:
                raise MissingAPIKeyError(
                    "GROQ_API_KEY is not configured. Gemini fallback is enabled, "
                    "but GEMINI_API_KEY is also missing."
                )
            _report_status(on_status, f"GROQ_API_KEY missing. Falling back to Gemini ({target_gemini_model})…")
        else:
            raise MissingAPIKeyError(
                "GROQ_API_KEY (or groq_api_keys) environment variable is missing or empty. "
                "Please configure GROQ_API_KEY in your .env file."
            )

    # Attempt Groq models in sequence
    groq_succeeded = False
    if groq_client is not None or groq_key:
        if groq_client is None:
            groq_client = get_groq_client(api_key=groq_key, timeout=timeout)

        for i, model_name in enumerate(groq_model_sequence):
            if i == 0:
                _report_status(on_status, f"Generating answer using Groq ({model_name})…")
            else:
                _report_status(on_status, f"Trying fallback Groq model ({model_name})…")

            try:
                answer = execute_groq_attempt(
                    client=groq_client,
                    model=model_name,
                    query=query,
                    results=results,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                _report_status(on_status, f"Answer generated successfully using Groq ({model_name}).")
                return RAGResponse(
                    answer=answer,
                    sources=list(results),
                    provider="Groq",
                    model=model_name,
                    is_fallback=False,
                    fallback_trace=fallback_trace,
                )
            except Exception as err:
                cleaned_err = _sanitize_error_message(err)
                is_retryable, reason = classify_groq_error(err)
                trace_entry = f"Groq ({model_name}) failed: {reason} [{cleaned_err}]"
                fallback_trace.append(trace_entry)

                if not is_retryable:
                    # Authentication or non-retryable error: do not retry other Groq models
                    _report_status(on_status, f"Groq {model_name} non-retryable error: {reason}.")
                    break

                if i + 1 < len(groq_model_sequence):
                    next_model = groq_model_sequence[i + 1]
                    _report_status(on_status, f"Primary model busy ({reason}). Trying another Groq model ({next_model})…")
                else:
                    _report_status(on_status, f"All configured Groq models failed ({reason}).")

    # If all Groq models failed, attempt optional Gemini fallback
    if gemini_allowed:
        gem_key = get_gemini_api_key(gemini_api_key)
        if not gem_key and gemini_client is None:
            fallback_trace.append("Gemini fallback skipped: GEMINI_API_KEY is not configured")
            _report_status(on_status, "Gemini fallback is enabled, but GEMINI_API_KEY is missing.")
            raise AllProvidersFailedError(
                f"All Groq models failed. Gemini fallback is enabled but GEMINI_API_KEY is missing. "
                f"Attempts: {'; '.join(fallback_trace)}"
            )

        _report_status(on_status, f"Groq models are currently unavailable. Switching to Gemini ({target_gemini_model})…")
        if gemini_client is None:
            gemini_client = get_gemini_client(api_key=gem_key)

        prompt = build_rag_prompt(query=query, results=results)
        try:
            answer = call_gemini_model(
                client=gemini_client,
                model=target_gemini_model,
                prompt=prompt,
                system_instruction=system_instruction,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            _report_status(on_status, f"Answer generated successfully using Gemini ({target_gemini_model}).")
            return RAGResponse(
                answer=answer,
                sources=list(results),
                provider="Gemini",
                model=target_gemini_model,
                is_fallback=False,
                fallback_trace=fallback_trace,
            )
        except Exception as gem_err:
            cleaned_gem = _sanitize_error_message(gem_err)
            fallback_trace.append(f"Gemini ({target_gemini_model}) failed: {cleaned_gem}")
            _report_status(on_status, "AI services are temporarily unavailable. Please try again later.")
            raise AllProvidersFailedError(
                f"AI services are temporarily unavailable. Please try again later. "
                f"({'; '.join(fallback_trace)})"
            )

    _report_status(on_status, "AI services are temporarily unavailable. Please try again later.")
    raise AllProvidersFailedError(
        f"AI services are temporarily unavailable. All configured Groq models failed. "
        f"({'; '.join(fallback_trace)})"
    )
