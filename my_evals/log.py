import logging

LOGGER = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_logger(name: str = __name__, level: str = "info"):
    setup_logging(level)
    return logging.getLogger(name)


def setup_logging(logging_level: str) -> None:
    level = LOGGING_LEVELS.get(logging_level.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
