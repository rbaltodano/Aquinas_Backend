import threading
import time
import unittest

from generation_coordinator import GenerationCoordinator


class GenerationCoordinatorTests(unittest.TestCase):
    def test_foreground_signals_active_background_and_runs_next(self) -> None:
        coordinator = GenerationCoordinator()
        background_started = threading.Event()
        foreground_finished = threading.Event()
        observed_preemption = threading.Event()

        def background_work() -> None:
            with coordinator.session("background") as lease:
                background_started.set()
                while not lease.cancel_event.wait(timeout=0.01):
                    pass
                observed_preemption.set()

        def foreground_work() -> None:
            with coordinator.session("foreground"):
                foreground_finished.set()

        background = threading.Thread(target=background_work)
        background.start()
        self.assertTrue(background_started.wait(timeout=1))

        foreground = threading.Thread(target=foreground_work)
        foreground.start()

        self.assertTrue(observed_preemption.wait(timeout=1))
        self.assertTrue(foreground_finished.wait(timeout=1))
        background.join(timeout=1)
        foreground.join(timeout=1)

    def test_background_waits_while_foreground_is_active(self) -> None:
        coordinator = GenerationCoordinator()
        foreground_started = threading.Event()
        release_foreground = threading.Event()
        background_started = threading.Event()

        def foreground_work() -> None:
            with coordinator.session("foreground"):
                foreground_started.set()
                release_foreground.wait(timeout=1)

        def background_work() -> None:
            with coordinator.session("background"):
                background_started.set()

        foreground = threading.Thread(target=foreground_work)
        foreground.start()
        self.assertTrue(foreground_started.wait(timeout=1))

        background = threading.Thread(target=background_work)
        background.start()
        time.sleep(0.05)
        self.assertFalse(background_started.is_set())

        release_foreground.set()
        self.assertTrue(background_started.wait(timeout=1))
        foreground.join(timeout=1)
        background.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
