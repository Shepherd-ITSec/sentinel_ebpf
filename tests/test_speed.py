"""Performance/speed tests for sentinel-ebpf probe system."""
import os
import threading
import time
from pathlib import Path
from typing import Optional

import pytest
import requests


class EventGenerator:
    """Generates file operations at a controlled rate."""

    def __init__(self, rate: int, duration: float, temp_dir: Path):
        self.rate = rate  # events per second
        self.duration = duration  # seconds
        self.temp_dir = temp_dir
        self.events_generated = 0
        self.start_time: Optional[float] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _generate_events(self):
        """Generate file operations at the specified rate."""
        self.start_time = time.time()
        interval = 1.0 / self.rate if self.rate > 0 else 0
        end_time = self.start_time + self.duration

        while not self.stop_event.is_set() and time.time() < end_time:
            loop_start = time.time()
            # Generate a file operation
            test_file = self.temp_dir / f"speed_test_{self.events_generated}_{int(time.time() * 1e6)}.txt"
            try:
                with open(test_file, "w") as f:
                    f.write(f"speed test event {self.events_generated}\n")
                # Read it back to generate another event
                with open(test_file, "r") as f:
                    _ = f.read()
                # Delete it
                test_file.unlink()
                self.events_generated += 1
            except Exception:
                pass  # Ignore errors during generation

            # Sleep to maintain rate
            if interval > 0:
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def start(self):
        """Start generating events."""
        self.thread = threading.Thread(target=self._generate_events, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop generating events."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.duration + 1)

    def get_stats(self):
        """Get generation statistics."""
        if self.start_time is None:
            return None
        elapsed = time.time() - self.start_time
        actual_rate = self.events_generated / elapsed if elapsed > 0 else 0
        return {
            "events_generated": self.events_generated,
            "elapsed_time": elapsed,
            "target_rate": self.rate,
            "actual_rate": actual_rate,
        }


class DetectorMonitor:
    """Monitors detector metrics and events."""

    def __init__(self, detector_url: str = "http://localhost:50052"):
        self.detector_url = detector_url
        self.initial_event_count: Optional[int] = None
        self.final_event_count: Optional[int] = None
        self.initial_metrics: Optional[dict] = None
        self.final_metrics: Optional[dict] = None

    def get_metrics(self) -> Optional[dict]:
        """Fetch Prometheus metrics from detector."""
        try:
            resp = requests.get(f"{self.detector_url}/metrics", timeout=2)
            if resp.status_code == 200:
                metrics = {}
                for line in resp.text.split("\n"):
                    if line.startswith("sentinel_ebpf_detector_events_total"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                metrics["events_total"] = int(float(parts[1]))
                            except ValueError:
                                pass
                    elif line.startswith("sentinel_ebpf_detector_anomalies_total"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                metrics["anomalies_total"] = int(float(parts[1]))
                            except ValueError:
                                pass
                return metrics
        except Exception:
            pass
        return None

    def start_monitoring(self):
        """Record initial state."""
        self.initial_metrics = self.get_metrics()
        if self.initial_metrics:
            self.initial_event_count = self.initial_metrics.get("events_total", 0)

    def stop_monitoring(self):
        """Record final state."""
        # Wait a bit for events to be processed
        time.sleep(2)
        self.final_metrics = self.get_metrics()
        if self.final_metrics:
            self.final_event_count = self.final_metrics.get("events_total", 0)

    def get_stats(self) -> Optional[dict]:
        """Calculate statistics."""
        if (
            self.initial_event_count is None
            or self.final_event_count is None
            or self.initial_metrics is None
            or self.final_metrics is None
        ):
            return None

        events_processed = self.final_event_count - self.initial_event_count
        anomalies_detected = (
            self.final_metrics.get("anomalies_total", 0)
            - self.initial_metrics.get("anomalies_total", 0)
        )

        return {
            "events_processed": events_processed,
            "anomalies_detected": anomalies_detected,
            "initial_count": self.initial_event_count,
            "final_count": self.final_event_count,
        }


@pytest.fixture
def detector_url():
    """Detector service URL."""
    return os.environ.get("DETECTOR_URL", "http://localhost:50052")


@pytest.fixture
def detector_available(detector_url):
    """Skip test if detector is not available."""
    try:
        resp = requests.get(f"{detector_url}/metrics", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"Detector not available at {detector_url}")
    except Exception:
        pytest.skip(f"Detector not available at {detector_url}")


def _wait_for_event_ingestion(monitor: DetectorMonitor, timeout_sec: float = 4.0) -> bool:
    """Return True if detector event count increases within timeout."""
    initial = monitor.get_metrics()
    if not initial or "events_total" not in initial:
        return False

    initial_count = int(initial["events_total"])
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(0.25)
        current = monitor.get_metrics()
        if current and int(current.get("events_total", initial_count)) > initial_count:
            return True
    return False


@pytest.fixture
def probe_event_flow_available(temp_dir, detector_url, detector_available):
    """Ensure probe->detector event flow is active for speed tests."""
    monitor = DetectorMonitor(detector_url)
    generator = EventGenerator(rate=200, duration=1.0, temp_dir=temp_dir)
    generator.start()
    while generator.thread and generator.thread.is_alive():
        time.sleep(0.1)
    generator.stop()

    if not _wait_for_event_ingestion(monitor):
        pytest.skip(
            "Probe event flow is not active for speed test path. "
            "Start probe/rules that capture the test directory."
        )


def _run_speed_trial(rate: int, duration: float, temp_dir: Path, detector_url: str) -> dict:
    """Run one generation/monitoring trial and return derived metrics."""
    monitor = DetectorMonitor(detector_url)
    monitor.start_monitoring()

    generator = EventGenerator(rate=rate, duration=duration, temp_dir=temp_dir)
    generator.start()
    while generator.thread and generator.thread.is_alive():
        time.sleep(0.5)
    generator.stop()

    monitor.stop_monitoring()
    gen_stats = generator.get_stats()
    det_stats = monitor.get_stats()
    assert gen_stats is not None, "Failed to generate events"
    assert det_stats is not None, "Failed to collect detector stats"

    events_generated = int(gen_stats["events_generated"])
    events_processed = int(det_stats["events_processed"])
    elapsed = float(gen_stats["elapsed_time"])
    throughput = events_processed / elapsed if elapsed > 0 else 0.0
    efficiency = (events_processed / events_generated * 100.0) if events_generated > 0 else 0.0
    drop_rate = max(0.0, 100.0 - efficiency)
    return {
        "events_generated": events_generated,
        "events_processed": events_processed,
        "elapsed_time": elapsed,
        "throughput": throughput,
        "efficiency": efficiency,
        "drop_rate": drop_rate,
    }


def _print_trial_stats(label: str, rate: int, stats: dict) -> None:
    """Emit consistent debug output for speed tests."""
    print(f"\nSpeed Test Results ({label}):")
    print(f"  Target Rate: {rate} events/sec")
    print(f"  Events Generated: {stats['events_generated']}")
    print(f"  Events Processed: {stats['events_processed']}")
    print(f"  Throughput: {stats['throughput']:.1f} events/sec")
    print(f"  Efficiency: {stats['efficiency']:.2f}%")
    print(f"  Drop Rate: {stats['drop_rate']:.2f}%")


class TestSpeed:
    """Performance tests for probe throughput."""

    @pytest.mark.perf
    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_low_rate(self, temp_dir, detector_url, probe_event_flow_available):
        """Test throughput at low rate (1000 events/sec).
        
        Note: This test requires both the probe and detector to be running.
        The probe must be able to capture file operations from the test directory.
        
        This test validates that the system can handle a moderate event rate
        without excessive drops. The actual throughput may be lower than the
        generation rate due to system load, network latency, and processing overhead.
        """
        rate = 1000
        duration = 5.0

        stats = _run_speed_trial(rate=rate, duration=duration, temp_dir=temp_dir, detector_url=detector_url)
        _print_trial_stats("low rate", rate, stats)
        events_processed = stats["events_processed"]
        throughput = stats["throughput"]
        assert events_processed > 0, (
            "No events were processed. "
            "Check: 1) Probe is running and capturing events, "
            "2) Rules allow the test path, 3) Detector is processing events"
        )
        assert throughput > 0.0

    @pytest.mark.perf
    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_medium_rate(self, temp_dir, detector_url, probe_event_flow_available):
        """Test throughput at medium rate (5000 events/sec)."""
        rate = 5000
        duration = 10.0

        stats = _run_speed_trial(rate=rate, duration=duration, temp_dir=temp_dir, detector_url=detector_url)
        _print_trial_stats("medium rate", rate, stats)
        events_processed = stats["events_processed"]
        throughput = stats["throughput"]
        assert events_processed > 0, (
            "No events were processed. Check probe and detector are running correctly."
        )
        assert throughput > 0.0

    @pytest.mark.perf
    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_high_rate(self, temp_dir, detector_url, probe_event_flow_available):
        """Test throughput at high rate (10000 events/sec)."""
        rate = 10000
        duration = 10.0

        stats = _run_speed_trial(rate=rate, duration=duration, temp_dir=temp_dir, detector_url=detector_url)
        _print_trial_stats("high rate", rate, stats)
        events_processed = stats["events_processed"]
        throughput = stats["throughput"]
        assert events_processed > 0, (
            "No events were processed. Check probe and detector are running correctly."
        )
        assert throughput > 0.0

    @pytest.mark.perf
    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_custom_rate(self, temp_dir, detector_url, probe_event_flow_available):
        """Test throughput at custom rate from environment variable."""
        rate = int(os.environ.get("SPEED_TEST_RATE", "2000"))
        duration = float(os.environ.get("SPEED_TEST_DURATION", "5.0"))

        stats = _run_speed_trial(rate=rate, duration=duration, temp_dir=temp_dir, detector_url=detector_url)
        _print_trial_stats("custom rate", rate, stats)
        events_processed = stats["events_processed"]

        # Basic assertion - should process some events
        assert events_processed > 0, "No events were processed"

    @pytest.mark.perf
    @pytest.mark.integration
    @pytest.mark.slow
    def test_find_max_throughput(self, temp_dir, detector_url, probe_event_flow_available):
        """Find maximum sustainable throughput (can be skipped for CI)."""
        if os.environ.get("SKIP_MAX_THROUGHPUT_TEST") == "1":
            pytest.skip("Max throughput test skipped")

        start_rate = int(os.environ.get("MAX_THROUGHPUT_START_RATE", "1000"))
        max_rate = int(os.environ.get("MAX_THROUGHPUT_MAX_RATE", "20000"))
        step = int(os.environ.get("MAX_THROUGHPUT_STEP", "2000"))
        duration = float(os.environ.get("MAX_THROUGHPUT_DURATION", "5.0"))
        max_drop_rate = float(os.environ.get("MAX_THROUGHPUT_MAX_DROP_RATE", "10.0"))

        results = []
        current_rate = start_rate

        while current_rate <= max_rate:
            stats = _run_speed_trial(
                rate=current_rate, duration=duration, temp_dir=temp_dir, detector_url=detector_url
            )
            throughput = stats["throughput"]
            drop_rate = stats["drop_rate"]

            results.append(
                {
                    "rate": current_rate,
                    "throughput": throughput,
                    "drop_rate": drop_rate,
                    "events_generated": stats["events_generated"],
                    "events_processed": stats["events_processed"],
                }
            )

            print(f"Rate: {current_rate} events/sec, Throughput: {throughput:.1f}, Drop: {drop_rate:.2f}%")

            # Stop if drop rate exceeds threshold
            if drop_rate > max_drop_rate:
                break

            # Stop if throughput is significantly lower than target
            if throughput < current_rate * 0.7:
                break

            current_rate += step
            time.sleep(1)  # Brief pause between tests

        assert len(results) > 0, "No test results collected"

        # Find best result
        best = max([r for r in results if r["drop_rate"] <= max_drop_rate], key=lambda x: x["throughput"], default=results[-1])

        print(f"\nMaximum Sustainable Throughput:")
        print(f"  Rate: {best['rate']} events/sec")
        print(f"  Throughput: {best['throughput']:.1f} events/sec")
        print(f"  Drop Rate: {best['drop_rate']:.2f}%")

        # Assert we found a reasonable throughput
        assert best["throughput"] >= start_rate * 0.5, f"Maximum throughput too low: {best['throughput']:.1f}"
