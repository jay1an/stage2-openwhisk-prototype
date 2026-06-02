from __future__ import annotations

import threading
import time
import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.stage5_control.jit_scheduler import JitScheduler, WarmupTask


class FireRecorder:
    def __init__(self):
        self._condition = threading.Condition()
        self.records: list[tuple[str, float, WarmupTask]] = []

    def callback(self, task: WarmupTask) -> None:
        with self._condition:
            self.records.append((task.task_key, time.monotonic(), task))
            self._condition.notify_all()

    def wait_for_count(self, count: int, timeout: float) -> list[tuple[str, float, WarmupTask]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self.records) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return list(self.records)

    def snapshot(self) -> list[tuple[str, float, WarmupTask]]:
        with self._condition:
            return list(self.records)


class TestJitScheduler(unittest.TestCase):
    def make_scheduler(self, recorder: FireRecorder) -> JitScheduler:
        scheduler = JitScheduler(recorder.callback)
        scheduler.start()
        self.addCleanup(scheduler.stop)
        return scheduler

    def assert_elapsed_close(
        self,
        start: float,
        actual: float,
        expected: float,
        tolerance: float = 0.08,
    ) -> None:
        elapsed = actual - start
        self.assertGreaterEqual(elapsed, expected - tolerance)
        self.assertLessEqual(elapsed, expected + tolerance)

    def test_single_task_fires_at_right_time(self) -> None:
        recorder = FireRecorder()
        scheduler = self.make_scheduler(recorder)
        start = time.monotonic()

        scheduler.schedule(WarmupTask("x", start + 0.2, "action_x"))

        records = recorder.wait_for_count(1, timeout=1.0)
        self.assertEqual([record[0] for record in records], ["x"])
        self.assert_elapsed_close(start, records[0][1], 0.2)

    def test_multiple_tasks_fire_in_time_order(self) -> None:
        recorder = FireRecorder()
        scheduler = self.make_scheduler(recorder)
        start = time.monotonic()

        scheduler.schedule(WarmupTask("task_3", start + 0.3, "action_3"))
        scheduler.schedule(WarmupTask("task_1", start + 0.1, "action_1"))
        scheduler.schedule(WarmupTask("task_2", start + 0.2, "action_2"))

        records = recorder.wait_for_count(3, timeout=1.2)
        self.assertEqual([record[0] for record in records], ["task_1", "task_2", "task_3"])

    def test_upsert_moves_task_earlier(self) -> None:
        recorder = FireRecorder()
        scheduler = self.make_scheduler(recorder)
        start = time.monotonic()

        scheduler.schedule(WarmupTask("x", start + 1.0, "action_x_old"))
        scheduler.schedule(WarmupTask("x", start + 0.2, "action_x_new"))

        records = recorder.wait_for_count(1, timeout=0.7)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], "x")
        self.assertEqual(records[0][2].action_name, "action_x_new")
        self.assert_elapsed_close(start, records[0][1], 0.2)
        time.sleep(0.35)
        self.assertEqual(len(recorder.snapshot()), 1)

    def test_upsert_moves_task_later(self) -> None:
        recorder = FireRecorder()
        scheduler = self.make_scheduler(recorder)
        start = time.monotonic()

        scheduler.schedule(WarmupTask("x", start + 0.2, "action_x_old"))
        scheduler.schedule(WarmupTask("x", start + 1.0, "action_x_new"))

        time.sleep(0.35)
        self.assertEqual(recorder.snapshot(), [])
        records = recorder.wait_for_count(1, timeout=1.0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], "x")
        self.assertEqual(records[0][2].action_name, "action_x_new")
        self.assert_elapsed_close(start, records[0][1], 1.0, tolerance=0.12)

    def test_cancel_removes_task(self) -> None:
        recorder = FireRecorder()
        scheduler = self.make_scheduler(recorder)
        start = time.monotonic()

        scheduler.schedule(WarmupTask("x", start + 0.3, "action_x"))
        time.sleep(0.1)
        scheduler.cancel("x")

        time.sleep(0.45)
        self.assertEqual(recorder.snapshot(), [])
        self.assertEqual(scheduler.pending_count(), 0)

    def test_no_duplicate_fires_under_upsert_churn(self) -> None:
        recorder = FireRecorder()
        scheduler = self.make_scheduler(recorder)
        start = time.monotonic()

        scheduler.schedule(WarmupTask("x", start + 0.8, "action_1"))
        scheduler.schedule(WarmupTask("x", start + 0.4, "action_2"))
        scheduler.schedule(WarmupTask("x", start + 0.7, "action_3"))
        scheduler.schedule(WarmupTask("x", start + 0.3, "action_final"))

        records = recorder.wait_for_count(1, timeout=0.8)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], "x")
        self.assertEqual(records[0][2].action_name, "action_final")
        self.assert_elapsed_close(start, records[0][1], 0.3)
        time.sleep(0.55)
        self.assertEqual(len(recorder.snapshot()), 1)

    def test_clean_shutdown_returns_promptly(self) -> None:
        recorder = FireRecorder()
        scheduler = JitScheduler(recorder.callback)
        scheduler.start()
        start = time.monotonic()
        scheduler.schedule(WarmupTask("future", start + 10.0, "action_future"))

        stop_start = time.monotonic()
        scheduler.stop()
        stop_elapsed = time.monotonic() - stop_start

        self.assertLess(stop_elapsed, 1.0)
        self.assertIsNotNone(scheduler._thread)
        self.assertFalse(scheduler._thread.is_alive())
        self.assertEqual(recorder.snapshot(), [])

    def test_pending_count_accuracy(self) -> None:
        recorder = FireRecorder()
        scheduler = self.make_scheduler(recorder)
        start = time.monotonic()

        scheduler.schedule(WarmupTask("first", start + 0.1, "action_first"))
        scheduler.schedule(WarmupTask("second", start + 1.0, "action_second"))
        scheduler.schedule(WarmupTask("third", start + 1.1, "action_third"))
        self.assertEqual(scheduler.pending_count(), 3)

        records = recorder.wait_for_count(1, timeout=0.6)
        self.assertEqual([record[0] for record in records], ["first"])
        self.assertEqual(scheduler.pending_count(), 2)
        scheduler.cancel("second")
        scheduler.cancel("third")
        self.assertEqual(scheduler.pending_count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
