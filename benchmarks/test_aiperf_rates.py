"""Native phase rates must include drain and reconcile to the audited population."""

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from aiperf_results import audit_aggregate_rates, validate
from test_endpoint import export, fixture


def phase_fixture(directory, slos=False):
    ordinary, raw, summary = fixture(directory, slos=slos)
    path = directory / "invocation.json"
    invocation = json.loads(path.read_text())
    invocation["client_timezone"] = "UTC"
    path.write_text(json.dumps(invocation))
    return ordinary, raw, summary


class AggregateRateTests(unittest.TestCase):
    def test_phase_rates_use_actual_usage_and_include_terminal_drain(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = phase_fixture(directory, slos=True)
            # Content is received much earlier than the phase ends; neither its
            # duration nor the requested cap is the finite-phase denominator/work.
            summary["benchmark_duration"] = {"avg": 2.1, "unit": "sec"}
            export(directory, ordinary, raw, summary)
            report = validate(directory)
            audit = report["aggregate_rate_audit"]
            self.assertEqual(audit["duration_ns_bounds"], [3_999_996_000, 4_000_004_000])
            self.assertEqual(
                audit["rates_per_second"],
                {"request_throughput": 0.5, "output_token_throughput": 1.0, "goodput": 0.5},
            )
            self.assertFalse(report["performance_metrics_audited"])

    def test_corrupt_rates_counts_units_and_content_only_denominator_reject(self):
        for mode in (
            "request-rate", "output-rate", "goodput", "osl-total", "units",
            "missing-rate", "content-denominator", "nan",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = phase_fixture(directory, slos=True)
                if mode == "request-rate":
                    summary["request_throughput"]["avg"] *= 2
                elif mode == "output-rate":
                    summary["output_token_throughput"]["avg"] *= 2
                elif mode == "goodput":
                    summary["goodput"]["avg"] *= 2
                elif mode == "osl-total":
                    summary["total_osl"]["avg"] += 1
                elif mode == "units":
                    summary["request_throughput"]["unit"] = "tokens/sec"
                elif mode == "missing-rate":
                    del summary["output_token_throughput"]
                elif mode == "nan":
                    summary["request_throughput"]["avg"] = float("nan")
                else:
                    for tag in ("request_throughput", "output_token_throughput", "goodput"):
                        summary[tag]["avg"] *= 4 / 2.1
                export(directory, ordinary, raw, summary)
                with self.assertRaises(ValueError):
                    validate(directory)

    def test_missing_timezone_leaves_rates_explicitly_unaudited(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = fixture(directory)
            # Old artifacts do not necessarily record the timezone used by the
            # naive datetime exporter. Do not infer it from today's environment.
            export(directory, ordinary, raw, summary)
            self.assertIsNone(validate(directory)["aggregate_rate_audit"])

    def test_invalid_or_reversed_phase_clocks_reject(self):
        for value in (
            None, "2026-01-01", "2026-01-01T00:00:04Z", "2025-12-31T23:59:59",
            "2026-01-01T00:00:00.000004",
        ):
            with self.subTest(end=value), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = phase_fixture(directory)
                summary["end_time"] = value
                export(directory, ordinary, raw, summary)
                with self.assertRaises(ValueError):
                    validate(directory)

    def test_binary64_and_microsecond_export_rounding_are_bounded_not_fitted(self):
        # Independently reproduce the native exporter at three epoch magnitudes,
        # including just below the documented 2**32-second limit.
        for start in (1_999, 1_767_225_600_000_000_499, 4_294_967_200_123_456_789):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = phase_fixture(directory)
                duration = 4_000_000_889
                for key, value in (("start_time", start), ("end_time", start + duration)):
                    summary[key] = (
                        datetime.fromtimestamp(value / 1_000_000_000, UTC)
                        .replace(tzinfo=None)
                        .isoformat()
                    )
                for tag, numerator in (
                    ("request_throughput", 2), ("output_token_throughput", 4)
                ):
                    summary[tag]["avg"] = numerator / (duration * (1 / 1_000_000_000))
                export(directory, ordinary, raw, summary)
                audit = validate(directory)["aggregate_rate_audit"]
                low, high = audit["duration_ns_bounds"]
                self.assertLessEqual(low, duration)
                self.assertGreaterEqual(high, duration)
                # The same timestamps do not excuse a materially wrong rate.
                summary["request_throughput"]["avg"] *= 1.01
                export(directory, ordinary, raw, summary)
                with self.assertRaisesRegex(ValueError, "aggregate rate mismatch"):
                    validate(directory)

    def test_zero_goodput_is_distinct_from_absent_undefined_goodput(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, summary = phase_fixture(Path(tmp))
            summary["goodput"] = {"avg": 0, "unit": "requests/sec"}
            defined = {"good_requests": 0, "violated_requests": 2}
            audit = audit_aggregate_rates(summary, 2, 4, defined, "UTC")
            self.assertEqual(audit["rates_per_second"]["goodput"], 0)
            summary["goodput"]["avg"] = 5e-324
            with self.assertRaisesRegex(ValueError, "aggregate rate mismatch"):
                audit_aggregate_rates(summary, 2, 4, defined, "UTC")
            undefined = {"good_requests": 0, "violated_requests": 0}
            summary["goodput"]["avg"] = 0
            with self.assertRaisesRegex(ValueError, "must be absent"):
                audit_aggregate_rates(summary, 2, 4, undefined, "UTC")
            del summary["goodput"]
            audit = audit_aggregate_rates(summary, 2, 4, undefined, "UTC")
            self.assertIsNone(audit["rates_per_second"]["goodput"])

    def test_invalid_protocol_cannot_attest_aggregate_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = phase_fixture(directory)
            raw[0]["responses"].pop()  # Missing DONE, despite native success.
            export(directory, ordinary, raw, summary)
            report = validate(directory)
            self.assertFalse(report["protocol_valid"])
            self.assertIsNone(report["aggregate_rate_audit"])

    def test_partial_failure_uses_whole_phase_and_no_success_only_slo_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = phase_fixture(directory, slos=True)
            ordinary[1].update(error={"type": "http", "message": "overloaded"}, metrics={})
            raw[1].update(status=429, responses=[], error={"message": "overloaded"})
            summary["request_count"]["avg"] = 1
            summary["error_request_count"] = {"avg": 1, "unit": "requests"}
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                summary[f"total_usage_{field}"]["avg"] //= 2
            summary["total_osl"]["avg"] = 2
            summary["request_throughput"]["avg"] = 0.25
            summary["output_token_throughput"]["avg"] = 0.5
            summary["good_request_count"]["avg"] = 1
            summary["good_request_fraction"]["avg"] = 0.5
            summary["goodput"]["avg"] = 0.25
            export(directory, ordinary, raw, summary)
            report = validate(directory)
            self.assertEqual(report["failed_requests"], 1)
            self.assertEqual(report["slo_attainment"]["attempted_fraction"], 0.5)
            self.assertEqual(
                report["aggregate_rate_audit"]["rates_per_second"],
                {"request_throughput": 0.25, "output_token_throughput": 0.5, "goodput": 0.25},
            )
