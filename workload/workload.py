import argparse
import signal
import sys
import time

_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def cpu_work(duration):
    """Burn CPU cycles continuously for the specified duration in seconds."""
    end_time = time.time() + duration
    while _running and time.time() < end_time:
        x = 0
        for i in range(500_000):
            x += i * i


def allocate_memory(memory_mb):
    """Allocate and touch memory pages to ensure actual resident set size (RSS)."""
    chunk_size = 1024 * 1024
    buffer = bytearray()
    for _ in range(memory_mb):
        buffer.extend(b"\xaa" * chunk_size)
    return buffer


def memory_work(duration, memory_mb):
    """Allocate specified memory in MB and hold resident for duration."""
    data = allocate_memory(memory_mb)
    end_time = time.time() + duration
    while _running and time.time() < end_time:
        time.sleep(min(0.5, max(0, end_time - time.time())))
    return data


def burst_work(duration, burst_active=2.0, burst_idle=1.0):
    """Alternate between active CPU bursts and idle rest cycles."""
    end_time = time.time() + duration
    while _running and time.time() < end_time:
        # Active phase
        active_end = min(time.time() + burst_active, end_time)
        while _running and time.time() < active_end:
            x = 0
            for i in range(500_000):
                x += i * i

        # Idle phase
        if _running and time.time() < end_time:
            sleep_time = min(burst_idle, end_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)


def mixed_work(duration, memory_mb):
    """Simultaneously exercise CPU and memory by holding allocated RAM while burning CPU."""
    data = allocate_memory(memory_mb)
    end_time = time.time() + duration
    while _running and time.time() < end_time:
        x = 0
        for i in range(500_000):
            x += i * i
    return data


def idle_work(duration):
    """Sleep for the specified duration."""
    end_time = time.time() + duration
    while _running and time.time() < end_time:
        time.sleep(min(0.5, max(0, end_time - time.time())))


def run_workload(mode, duration=60, memory_mb=256, burst_active=2.0, burst_idle=1.0):
    """Dispatch and execute the requested workload mode."""
    if mode == "cpu":
        cpu_work(duration)
    elif mode == "memory":
        memory_work(duration, memory_mb)
    elif mode == "burst":
        burst_work(duration, burst_active=burst_active, burst_idle=burst_idle)
    elif mode == "mixed":
        mixed_work(duration, memory_mb)
    elif mode == "idle":
        idle_work(duration)
    else:
        raise ValueError(f"Unknown workload mode: {mode}")


def build_parser():
    parser = argparse.ArgumentParser(description="NeuroScale Controlled Workload Generator")

    parser.add_argument(
        "--mode",
        choices=["idle", "cpu", "memory", "burst", "mixed"],
        default="cpu",
        help="Workload mode to execute (idle, cpu, memory, burst, mixed)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration of the workload in seconds"
    )

    parser.add_argument(
        "--memory",
        type=int,
        default=256,
        help="Memory to allocate in MB for memory and mixed modes"
    )

    parser.add_argument(
        "--burst-on",
        "--burst-active",
        dest="burst_active",
        type=float,
        default=2.0,
        help="Active burst duration in seconds for burst mode"
    )

    parser.add_argument(
        "--burst-off",
        "--burst-idle",
        dest="burst_idle",
        type=float,
        default=1.0,
        help="Idle rest duration in seconds for burst mode"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(
        f"NeuroScale workload started: "
        f"mode={args.mode}, duration={args.duration}, memory={args.memory}MB",
        flush=True
    )

    run_workload(
        mode=args.mode,
        duration=args.duration,
        memory_mb=args.memory,
        burst_active=args.burst_active,
        burst_idle=args.burst_idle
    )

    print("NeuroScale workload finished", flush=True)


if __name__ == "__main__":
    main()