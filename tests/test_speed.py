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


class TestSpeed:
    """Performance tests for probe throughput."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_low_rate(self, temp_dir, detector_url, detector_available):
        """Test throughput at low rate (1000 events/sec).
        
        Note: This test requires both the probe and detector to be running.
        The probe must be able to capture file operations from the test directory.
        
        This test validates that the system can handle a moderate event rate
        without excessive drops. The actual throughput may be lower than the
        generation rate due to system load, network latency, and processing overhead.
        """
        rate = 1000
        duration = 5.0

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

        events_processed = det_stats["events_processed"]
        events_generated = gen_stats["events_generated"]
        throughput = events_processed / gen_stats["elapsed_time"] if gen_stats["elapsed_time"] > 0 else 0
        efficiency = (events_processed / events_generated * 100) if events_generated > 0 else 0

        # Print results for debugging
        print(f"\nSpeed Test Results (low rate):")
        print(f"  Target Rate: {rate} events/sec")
        print(f"  Events Generated: {events_generated}")
        print(f"  Events Processed: {events_processed}")
        print(f"  Throughput: {throughput:.1f} events/sec")
        print(f"  Efficiency: {efficiency:.2f}%")
        print(f"  Drop Rate: {100 - efficiency:.2f}%")

        # Basic check: some events should be processed
        assert events_processed > 0, (
            "No events were processed. "
            "Check: 1) Probe is running and capturing events, "
            "2) Rules allow the test path, 3) Detector is processing events"
        )
        
        # At low rate, we expect reasonable efficiency (at least 50%)
        # Lower efficiency indicates system is overloaded or probe/detector issues
        assert efficiency >= 50.0, (
            f"Efficiency too low: {efficiency:.2f}% (expected >=50% at {rate} events/sec). "
            f"This suggests the system is overloaded or there's a bottleneck. "
            f"Generated: {events_generated}, Processed: {events_processed}, "
            f"Throughput: {throughput:.1f} events/sec"
        )
        
        # Throughput should be reasonable - at least 400 events/sec at low rate
        # (allows for some overhead but ensures system is functioning)
        assert throughput >= 400.0, (
            f"Throughput too low: {throughput:.1f} events/sec (expected >=400 at {rate} events/sec). "
            f"This indicates a significant bottleneck in the system."
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_medium_rate(self, temp_dir, detector_url, detector_available):
        """Test throughput at medium rate (5000 events/sec)."""
        rate = 5000
        duration = 10.0

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

        events_processed = det_stats["events_processed"]
        events_generated = gen_stats["events_generated"]
        throughput = events_processed / gen_stats["elapsed_time"]
        drop_rate = ((events_generated - events_processed) / events_generated * 100) if events_generated > 0 else 0

        efficiency = (events_processed / events_generated * 100) if events_generated > 0 else 0
        
        # Print results for debugging
        print(f"\nSpeed Test Results (medium rate):")
        print(f"  Target Rate: {rate} events/sec")
        print(f"  Events Generated: {events_generated}")
        print(f"  Events Processed: {events_processed}")
        print(f"  Throughput: {throughput:.1f} events/sec")
        print(f"  Efficiency: {efficiency:.2f}%")
        print(f"  Drop Rate: {drop_rate:.2f}%")

        # Basic check: some events should be processed
        assert events_processed > 0, (
            "No events were processed. Check probe and detector are running correctly."
        )
        
        # At medium rate, expect at least 30% efficiency
        # Higher drop rates are expected as we approach system limits
        assert efficiency >= 30.0, (
            f"Efficiency too low: {efficiency:.2f}% (expected >=30% at {rate} events/sec). "
            f"System appears overloaded. Generated: {events_generated}, "
            f"Processed: {events_processed}, Throughput: {throughput:.1f} events/sec"
        )
        
        # Throughput should be at least 1000 events/sec at medium rate
        # (allows for significant overhead but ensures basic functionality)
        assert throughput >= 1000.0, (
            f"Throughput too low: {throughput:.1f} events/sec (expected >=1000 at {rate} events/sec). "
            f"System bottleneck detected."
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_high_rate(self, temp_dir, detector_url, detector_available):
        """Test throughput at high rate (10000 events/sec)."""
        rate = 10000
        duration = 10.0

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

        events_processed = det_stats["events_processed"]
        events_generated = gen_stats["events_generated"]
        throughput = events_processed / gen_stats["elapsed_time"]
        drop_rate = ((events_generated - events_processed) / events_generated * 100) if events_generated > 0 else 0

        efficiency = (events_processed / events_generated * 100) if events_generated > 0 else 0
        
        # Print results for debugging
        print(f"\nSpeed Test Results (high rate):")
        print(f"  Target Rate: {rate} events/sec")
        print(f"  Events Generated: {events_generated}")
        print(f"  Events Processed: {events_processed}")
        print(f"  Throughput: {throughput:.1f} events/sec")
        print(f"  Efficiency: {efficiency:.2f}%")
        print(f"  Drop Rate: {drop_rate:.2f}%")

        # Basic check: some events should be processed
        assert events_processed > 0, (
            "No events were processed. Check probe and detector are running correctly."
        )
        
        # At high rate, expect at least 20% efficiency
        # Significant drops are expected when pushing system limits
        assert efficiency >= 20.0, (
            f"Efficiency too low: {efficiency:.2f}% (expected >=20% at {rate} events/sec). "
            f"System is severely overloaded. Generated: {events_generated}, "
            f"Processed: {events_processed}, Throughput: {throughput:.1f} events/sec"
        )
        
        # Throughput should be at least 1500 events/sec at high rate
        # (allows for heavy overhead but ensures system is still processing)
        assert throughput >= 1500.0, (
            f"Throughput too low: {throughput:.1f} events/sec (expected >=1500 at {rate} events/sec). "
            f"System bottleneck detected."
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_throughput_custom_rate(self, temp_dir, detector_url, detector_available):
        """Test throughput at custom rate from environment variable."""
        rate = int(os.environ.get("SPEED_TEST_RATE", "2000"))
        duration = float(os.environ.get("SPEED_TEST_DURATION", "5.0"))

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

        events_processed = det_stats["events_processed"]
        events_generated = gen_stats["events_generated"]
        throughput = events_processed / gen_stats["elapsed_time"]
        efficiency = (events_processed / events_generated * 100) if events_generated > 0 else 0

        # Print results for debugging
        print(f"\nSpeed Test Results:")
        print(f"  Target Rate: {rate} events/sec")
        print(f"  Events Generated: {events_generated}")
        print(f"  Events Processed: {events_processed}")
        print(f"  Throughput: {throughput:.1f} events/sec")
        print(f"  Efficiency: {efficiency:.2f}%")

        # Basic assertion - should process some events
        assert events_processed > 0, "No events were processed"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_find_max_throughput(self, temp_dir, detector_url, detector_available):
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
            monitor = DetectorMonitor(detector_url)
            monitor.start_monitoring()

            generator = EventGenerator(rate=current_rate, duration=duration, temp_dir=temp_dir)
            generator.start()

            while generator.thread and generator.thread.is_alive():
                time.sleep(0.5)

            generator.stop()
            monitor.stop_monitoring()

            gen_stats = generator.get_stats()
            det_stats = monitor.get_stats()

            if gen_stats and det_stats:
                events_processed = det_stats["events_processed"]
                events_generated = gen_stats["events_generated"]
                throughput = events_processed / gen_stats["elapsed_time"]
                drop_rate = ((events_generated - events_processed) / events_generated * 100) if events_generated > 0 else 0

                results.append(
                    {
                        "rate": current_rate,
                        "throughput": throughput,
                        "drop_rate": drop_rate,
                        "events_generated": events_generated,
                        "events_processed": events_processed,
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
