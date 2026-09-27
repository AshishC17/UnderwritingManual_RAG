"""Sanitise arguments before tracing and cached function execution."""
from functools import wraps
from inspect import signature

from langsmith import traceable as _traceable

from src.guardrails.privacy import sanitize, trace_payload
from src.guardrails.core import cache_directory


def traceable(**options):
    def decorate(function):
        @wraps(function)
        def execute(*args, **kwargs):
            try:
                return sanitize(function(*args, **kwargs), "function_output")
            except Exception as exc:
                # Traces include exception strings, which provider/parser
                # exceptions can otherwise fill with raw response content.
                exc.args = (f"{type(exc).__name__}: operation failed",)
                raise
        traced = _traceable(**options, process_inputs=trace_payload,
                            process_outputs=trace_payload)(execute)
        sig = signature(function)

        @wraps(function)
        def boundary(*args, **kwargs):
            # LangSmith's extra kwargs belong to tracing, not the function.
            extra = {k: v for k, v in kwargs.items() if k == "langsmith_extra"}
            arguments = sig.bind(*args, **{k: v for k, v in kwargs.items() if k not in extra})
            arguments.apply_defaults()
            if function.__name__ == "embed_document":
                from src.guardrails.checks import evidence_check
                evidence_check([{"text": text} for text in arguments.arguments["texts"]])
            if "cache_dir" in arguments.arguments:
                arguments.arguments["cache_dir"] = cache_directory(arguments.arguments["cache_dir"])
            for key, value in arguments.arguments.items():
                arguments.arguments[key] = sanitize(value, "before_trace")
            return traced(*arguments.args, **arguments.kwargs, **extra)
        return boundary
    return decorate
