import json
from datetime import datetime

def parse_timestamp(ts_str):
    # Keep up to microseconds only (nanoseconds aren't supported by datetime)
    return datetime.strptime(ts_str[:26], "%Y-%m-%d %H:%M:%S.%f")

def analyze_worker_log(log_path):
    invocations = {}

    with open(log_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line)
                msg = log.get("fields", {}).get("message", "")
                ts = parse_timestamp(log["timestamp"])
                tid = log.get("fields", {}).get("tid", None)

                if not tid:
                    continue

                if tid not in invocations:
                    invocations[tid] = {}

                if msg == "Handling invocation request":
                    invocations[tid]["start"] = ts
                elif msg == "Container cold start completed":
                    invocations[tid]["end"] = ts
            except Exception as e:
                print(f"Skipping malformed line: {e}")

    print("📊 Cold Start Times (seconds):\n")
    for tid, data in invocations.items():
        if "start" in data and "end" in data:
            duration = (data["end"] - data["start"]).total_seconds()
            print(f"TID {tid}: {duration:.6f} seconds")
        else:
            print(f"TID {tid}: Incomplete log entries")

# Run the analysis
analyze_worker_log("/tmp/iluvatar/logs/worker.log")