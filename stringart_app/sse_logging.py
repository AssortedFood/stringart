# stringart_app/sse_logging.py

import logging

class SSELogHandler(logging.Handler):
    """
    Logging handler that writes log records into a per-job log list.
    """
    def __init__(self, job_id: str, job_logs: dict[str, list[str]]):
        super().__init__()
        self.job_id = job_id
        self.job_logs = job_logs

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.job_logs.setdefault(self.job_id, []).append(msg)

def create_sse_logger(job_id: str,
                      job_logs: dict[str, list[str]],
                      level: int = logging.DEBUG,
                      fmt: str = "%(message)s") -> logging.Logger:
    """
    Returns a logger configured with an SSELogHandler writing into job_logs[job_id].
    """
    logger = logging.getLogger(f"sse.{job_id}")
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicates
    handler = SSELogHandler(job_id, job_logs)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger
