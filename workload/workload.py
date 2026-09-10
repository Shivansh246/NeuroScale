import time

def cpu_work():
    while True:
        x = 0
        for i in range(10_000_000):
            x += i * i

def main():
    print("NeuroScale workload started")

    while True:
        cpu_work()
        time.sleep(1)

if __name__ == "__main__":
    main()
