import logging
import sys

import structlog


#sets up structlog so every log line comes out as json with consistent fields (timestamp,
#level, request id, etc). matters once this runs in docker and logs get shipped somewhere
#for searching, plain text logs are painful to query at that point
def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


#grabs a structlog logger by name, just a thin wrapper so callers don't import structlog directly
def get_logger(name: str = "deepequity") -> structlog.types.FilteringBoundLogger:
    return structlog.get_logger(name)
