import tempfile
import unittest
from pathlib import Path

from host_cpu_guard import (
    _read_cpu_ticks,
    calculate_interval,
    find_qemu_pids,
    format_cpu_list,
    parse_cpu_list,
)


class HostCpuGuardTest(unittest.TestCase):
    def test_cpu_list_round_trip(self) -> None:
        cpus = parse_cpu_list("1-3,7,9-10")
        self.assertEqual(cpus, {1, 2, 3, 7, 9, 10})
        self.assertEqual(format_cpu_list(cpus), "1-3,7,9-10")

    def test_interval_subtracts_qemu_from_reserved_cpu_busy_time(self) -> None:
        previous = {
            "cpu": {1: (100, 60), 2: (100, 50)},
            "qemu": {
                "10": {"ticks": 100, "start_time": 1},
                "20": {"ticks": 200, "start_time": 2},
            },
        }
        current = {
            "cpu": {1: (200, 120), 2: (200, 120)},
            "qemu": {
                "10": {"ticks": 150, "start_time": 1},
                "20": {"ticks": 260, "start_time": 2},
            },
        }
        result = calculate_interval(previous, current)
        self.assertEqual(result["busy_ticks"], 130)
        self.assertEqual(result["qemu_ticks"], 110)
        self.assertEqual(result["foreign_busy_ticks"], 20)
        self.assertAlmostEqual(result["foreign_busy_fraction"], 0.1)

    def test_proc_stat_does_not_double_count_guest_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stat").write_text("cpu12 100 10 20 30 5 2 3 1 40 4\n")
            self.assertEqual(_read_cpu_ticks({12}, root), {12: (171, 130)})

    def test_qemu_identity_change_is_rejected(self) -> None:
        previous = {
            "cpu": {1: (100, 60)},
            "qemu": {"10": {"ticks": 100, "start_time": 1}},
        }
        current = {
            "cpu": {1: (200, 120)},
            "qemu": {"10": {"ticks": 150, "start_time": 2}},
        }
        with self.assertRaises(RuntimeError):
            calculate_interval(previous, current)

    def test_discovery_requires_qemu_executable_not_argument_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, argv in {
                "10": ["/usr/bin/qemu-system-x86_64", "-name", "guest=omniflow"],
                "20": ["python3", "check.py", "qemu-system-x86_64", "omniflow"],
            }.items():
                process = root / pid
                process.mkdir()
                (process / "cmdline").write_bytes(
                    b"\0".join(item.encode() for item in argv) + b"\0"
                )
            self.assertEqual(find_qemu_pids("omniflow", root), [10])


if __name__ == "__main__":
    unittest.main()
