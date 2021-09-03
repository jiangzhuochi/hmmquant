__all__ = ["signal_logger", "rr_logger", "stdout_logger"]

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _logger

if TYPE_CHECKING:
    from loguru import Logger


LOG_PATH = Path(".") / "logs"

LOG_PATH.mkdir(parents=True, exist_ok=True)
LOG_FORMAT = (
    "<level><v>{level:<8}</v>"
    " [{time:YYYY-MM-DD} {time:HH:mm:ss.SSS} <d>{module}:{name}:{line}</d>]</level>"
    " {message}"
)
LOG_LEVEL = "TRACE"


def make_filter(name):
    def filter(record):
        return record["extra"].get("name") == name

    return filter


logger: "Logger" = _logger.opt(colors=True)
logger.remove()


logger.add(
    LOG_PATH / "signal.log",
    format="{message}",
    level=LOG_LEVEL,
    encoding="utf-8",
    filter=make_filter("signal"),
)
logger.add(
    LOG_PATH / "rr.log",
    format="{message}",
    level=LOG_LEVEL,
    encoding="utf-8",
    filter=make_filter("rr"),
)
logger.add(
    sys.stdout,
    format="{message}",
    level=LOG_LEVEL,
    filter=make_filter("stdout"),
)

# 记录信号
signal_logger = logger.bind(name="signal")
# 记录收益
rr_logger = logger.bind(name="rr")
# 标准输出
stdout_logger = logger.bind(name="stdout")

if __name__ == "__main__":
    signal_logger.info("sig")
    rr_logger.info("rr")
    stdout_logger.info("stdout")
