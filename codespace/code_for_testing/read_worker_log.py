import json
from datetime import datetime

def observed_wait_times(file_path):
    queue_data = []

    # Step 1: Parse queue snapshots
    with open(file_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line)
                if log.get("fields", {}).get("message") != "current load status":
                    continue

                ts_str = log["timestamp"]
                ts = datetime.strptime(ts_str[:26], "%Y-%m-%d %H:%M:%S.%f")
                status = json.loads(log["fields"]["status"])
                qlen = status.get("cpu_queue_len")

                if qlen is not None:
                    queue_data.append((ts, qlen))
            except Exception:
                continue

    if not queue_data:
        print("No valid queue entries found.")
        return

    base_time = next((ts for ts, q in queue_data if q == 128), queue_data[0][0])
    t_base = None
    for i in range(len(queue_data) - 2, -1, -1):
        t1, q1 = queue_data[i]
        t0, q0 = queue_data[i + 1]
        if q1 != 0:
            t_base = t0
            break
    t_curr = queue_data[0][0]
    for i in range(1, len(queue_data)):
        t0, q0 = queue_data[i - 1]
        t1, q1 = queue_data[i]
        if q1 < q0:
            t_curr = t1
        if q1 == 0:
            wait_ms = 0
            rel_time = 0
        else:
            wait_ms = (t_base-t_curr).total_seconds() * 1000
        # wait_ms = (t1 - t0).total_seconds() * 1000
            rel_time = int((t1 - base_time).total_seconds() * 1000)
        if wait_ms < 0 or q1 == 0:
            wait_ms = 0
        print(f"{q1},{rel_time},{round(wait_ms, 2)}")

if __name__ == "__main__":
    observed_wait_times("/tmp/iluvatar/logs/worker.log")