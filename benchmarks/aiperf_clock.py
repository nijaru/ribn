"""UTC lifecycle timestamps serialized by AIPerf 0.13's JSON exporter."""

import re
from datetime import datetime

from endpoint_config import require

# Integer ns -> binary64 seconds -> microsecond datetime, below 2**32 seconds.
# This bounds serialization only, not wall/monotonic or worker/controller alignment.
TIMESTAMP_UNCERTAINTY_NS = 2000


def exported_epoch_ns(value, field):
    require(
        isinstance(value, str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?",
            value,
        ),
        f"missing/invalid UTC phase {field}",
    )
    elapsed = datetime.fromisoformat(value) - datetime(1970, 1, 1)
    epoch_ns = (elapsed.days * 86400 + elapsed.seconds) * 1_000_000_000
    epoch_ns += elapsed.microseconds * 1000
    require(
        0 <= epoch_ns < (1 << 32) * 1_000_000_000, "phase epoch outside supported range"
    )
    return epoch_ns
