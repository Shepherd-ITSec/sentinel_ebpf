"""LID-DS-style performance reporting for scored event streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PerformanceEvent:
  predicted_anomaly: bool
  expected_anomaly: bool | None
  recording_name: str | None = None
  ts_unix_nano: int | None = None


class PerformanceMeasurement:
  """Accumulate LID-DS-style performance metrics from labeled event streams."""

  def __init__(self) -> None:
    self._threshold = 0.0
    self._current_recording_name: str | None = None
    self._current_recording_has_exploit = False
    self._current_recording_exploit_started = False
    self._current_recording_false_alarm = False
    self._current_recording_correct_alarm = False
    self._current_recording_exploit_time_s: float | None = None
    self._current_recording_detection_time_s: float | None = None
    self._current_cfp_stream_exploits = 0
    self._current_cfp_stream_normal = 0
    self._cfp_counter_wait_exploits = False
    self._cfp_counter_wait_normal = False
    self._exploit_anomaly_score_count = 0
    self._normal_score_count = 0
    self._first_syscall_of_cfp_list_exploits: list[int] = []
    self._last_syscall_of_cfp_list_exploits: list[int] = []
    self._first_syscall_of_cfp_list_normal: list[int] = []
    self._last_syscall_of_cfp_list_normal: list[int] = []

    self._fp = 0
    self._tp = 0
    self._tn = 0
    self._fn = 0
    self._correct_alarm_count = 0
    self._false_alarm_count = 0
    self._exploit_count = 0
    self._events_total = 0
    self._events_flagged = 0
    self._events_gt_anomalous = 0
    self._cfp_count_exploits = 0
    self._cfp_count_normal = 0
    self._detection_delays_s: list[float] = []
    self.result: dict[str, Any] | None = None

  def set_threshold(self, threshold: float) -> None:
    self._threshold = float(threshold)

  def add_event(self, event: PerformanceEvent) -> None:
    if event.expected_anomaly is None:
      return
    recording_name = (event.recording_name or "").strip() or None
    if recording_name != self._current_recording_name:
      self._start_recording(recording_name)

    self._analyze_event(event)

  def finalize(self) -> None:
    self._finish_recording()

  def get_cfp_indices(self) -> tuple[list[int], list[int], list[int], list[int]]:
    return (
      list(self._first_syscall_of_cfp_list_exploits),
      list(self._last_syscall_of_cfp_list_exploits),
      list(self._first_syscall_of_cfp_list_normal),
      list(self._last_syscall_of_cfp_list_normal),
    )

  def get_results(self) -> dict[str, Any]:
    self.finalize()
    detection_rate = self._correct_alarm_count / self._exploit_count if self._exploit_count else 0.0
    precision_syscall = self._tp / (self._tp + self._fp) if (self._tp + self._fp) else 0.0
    recall_syscall = self._tp / (self._tp + self._fn) if (self._tp + self._fn) else 0.0
    f1_syscall = (
      2 * (precision_syscall * recall_syscall) / (precision_syscall + recall_syscall)
      if (precision_syscall + recall_syscall)
      else 0.0
    )
    precision_cfa = (
      self._correct_alarm_count / (self._correct_alarm_count + self._cfp_count_normal + self._cfp_count_exploits)
      if (self._correct_alarm_count + self._cfp_count_normal + self._cfp_count_exploits)
      else 0.0
    )
    precision_sys = (
      self._correct_alarm_count / (self._correct_alarm_count + self._fp)
      if (self._correct_alarm_count + self._fp)
      else 0.0
    )
    f1_cfa = (
      2 * (precision_cfa * detection_rate) / (precision_cfa + detection_rate)
      if (precision_cfa + detection_rate)
      else 0.0
    )
    mean_detection_delay_s = (
      sum(self._detection_delays_s) / len(self._detection_delays_s)
      if self._detection_delays_s
      else None
    )
    percent_events_flagged = 100.0 * self._events_flagged / self._events_total if self._events_total else 0.0
    percent_events_gt_anomalous = 100.0 * self._events_gt_anomalous / self._events_total if self._events_total else 0.0

    self.result = {
      "threshold": self._threshold,
      "false_positives": self._fp,
      "true_positives": self._tp,
      "true_negatives": self._tn,
      "false_negatives": self._fn,
      "correct_alarm_count": self._correct_alarm_count,
      "false_alarm_count": self._false_alarm_count,
      "exploit_count": self._exploit_count,
      "detection_rate": detection_rate,
      "percent_events_flagged": percent_events_flagged,
      "percent_events_gt_anomalous": percent_events_gt_anomalous,
      "consecutive_false_positives_normal": self._cfp_count_normal,
      "consecutive_false_positives_exploits": self._cfp_count_exploits,
      "recall": detection_rate,
      "precision_syscall": precision_syscall,
      "recall_syscall": recall_syscall,
      "f1_syscall": f1_syscall,
      "precision_with_cfa": precision_cfa,
      "precision_with_syscalls": precision_sys,
      "f1_cfa": f1_cfa,
      "mean_detection_delay_s": mean_detection_delay_s,
      "metric_levels": {
        "true_positives": "syscall",
        "false_positives": "syscall",
        "true_negatives": "syscall",
        "false_negatives": "syscall",
        "percent_events_flagged": "syscall",
        "percent_events_gt_anomalous": "syscall",
        "precision_syscall": "syscall",
        "recall_syscall": "syscall",
        "f1_syscall": "syscall",
        "correct_alarm_count": "recording",
        "false_alarm_count": "recording",
        "exploit_count": "recording",
        "detection_rate": "recording",
        "recall": "recording",
        "consecutive_false_positives_normal": "recording",
        "consecutive_false_positives_exploits": "recording",
        "precision_with_cfa": "recording",
        "precision_with_syscalls": "mixed",
        "f1_cfa": "recording",
        "mean_detection_delay_s": "recording",
      },
    }
    return self.result

  def _start_recording(self, recording_name: str | None) -> None:
    self._finish_recording()
    self._current_recording_name = recording_name
    self._current_recording_has_exploit = False
    self._current_recording_exploit_started = False
    self._current_recording_false_alarm = False
    self._current_recording_correct_alarm = False
    self._current_recording_exploit_time_s = None
    self._current_recording_detection_time_s = None
    self._current_cfp_stream_exploits = 0
    self._current_cfp_stream_normal = 0
    self._cfp_counter_wait_exploits = False
    self._cfp_counter_wait_normal = False
    self._exploit_anomaly_score_count = 0
    self._normal_score_count = 0

  def _finish_recording(self) -> None:
    if self._current_recording_name is None:
      return
    if self._current_recording_has_exploit:
      self._cfp_end_exploits()
      if self._current_recording_false_alarm:
        self._false_alarm_count += 1
    else:
      self._cfp_end_normal()
      if self._current_recording_false_alarm:
        self._false_alarm_count += 1
    self._current_recording_name = None

  def _analyze_event(self, event: PerformanceEvent) -> None:
    assert event.expected_anomaly is not None
    predicted_anomaly = bool(event.predicted_anomaly)
    expected_anomaly = bool(event.expected_anomaly)

    self._events_total += 1
    if predicted_anomaly:
      self._events_flagged += 1
    if expected_anomaly:
      self._events_gt_anomalous += 1

    if expected_anomaly:
      self._current_recording_has_exploit = True
      if not self._current_recording_exploit_started:
        self._current_recording_exploit_started = True
        self._exploit_count += 1
        if event.ts_unix_nano is not None:
          self._current_recording_exploit_time_s = float(event.ts_unix_nano) * 1e-9

    if self._current_recording_has_exploit:
      self._exploit_anomaly_score_count += 1
      if predicted_anomaly:
        if not expected_anomaly:
          self._fp += 1
          self._current_cfp_stream_exploits += 1
          self._cfp_start_exploits()
          self._current_recording_false_alarm = True
        else:
          self._cfp_end_exploits()
          self._tp += 1
          if not self._current_recording_correct_alarm:
            self._current_recording_correct_alarm = True
            self._correct_alarm_count += 1
            if event.ts_unix_nano is not None:
              self._current_recording_detection_time_s = float(event.ts_unix_nano) * 1e-9
              if self._current_recording_exploit_time_s is not None:
                self._detection_delays_s.append(
                  self._current_recording_detection_time_s - self._current_recording_exploit_time_s
                )
      else:
        self._cfp_end_exploits()
        if expected_anomaly:
          self._fn += 1
        else:
          self._tn += 1
      return

    self._normal_score_count += 1
    if predicted_anomaly:
      self._fp += 1
      self._current_cfp_stream_normal += 1
      self._cfp_start_normal()
      self._current_recording_false_alarm = True
    else:
      self._cfp_end_normal()
      self._tn += 1

  def _cfp_start_exploits(self) -> None:
    if not self._cfp_counter_wait_exploits:
      self._first_syscall_of_cfp_list_exploits.append(self._exploit_anomaly_score_count)
      self._cfp_counter_wait_exploits = True

  def _cfp_end_exploits(self) -> None:
    if self._cfp_counter_wait_exploits and self._current_cfp_stream_exploits > 0:
      self._current_cfp_stream_exploits = 0
      self._cfp_count_exploits += 1
      self._last_syscall_of_cfp_list_exploits.append(self._exploit_anomaly_score_count)
      self._cfp_counter_wait_exploits = False

  def _cfp_start_normal(self) -> None:
    if not self._cfp_counter_wait_normal:
      self._first_syscall_of_cfp_list_normal.append(self._normal_score_count)
      self._cfp_counter_wait_normal = True

  def _cfp_end_normal(self) -> None:
    if self._cfp_counter_wait_normal and self._current_cfp_stream_normal > 0:
      self._current_cfp_stream_normal = 0
      self._cfp_count_normal += 1
      self._last_syscall_of_cfp_list_normal.append(self._normal_score_count)
      self._cfp_counter_wait_normal = False
