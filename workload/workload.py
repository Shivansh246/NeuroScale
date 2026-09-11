import argparse
import time


def cpu_work(duration):
    end_time = time.time() + duration

    while time.time() < end_time:
        x = 0
        for i in range(1_000_000):
            x += i * i


def memory_work(duration, memory_mb):
    data = bytearray(memory_mb * 1024 * 1024)

    end_time = time.time() + duration

    while time.time() < end_time:
        time.sleep(1)

    # Keep the allocated memory alive until the workload finishes.
    return data


def idle(duration):
    time.sleep(duration)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["idle", "cpu", "memory"],
        default="cpu"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60
    )

    parser.add_argument(
        "--memory",
        type=int,
        default=256,
        help="Memory to allocate in MB for memory mode"
    )

    args = parser.parse_args()

    print(
        f"NeuroScale workload started: "
        f"mode={args.mode}, duration={args.duration}",
        flush=True
    )

    if args.mode == "cpu":
        cpu_work(args.duration)

    elif args.mode == "memory":
        memory_work(args.duration, args.memory)

    elif args.mode == "idle":
        idle(args.duration)

    print("NeuroScale workload finished", flush=True)


if __name__ == "__main__":
    main()