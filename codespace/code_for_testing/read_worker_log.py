import json
from datetime import datetime

def parse_log_wait_time(file_path):
    queue_data = []

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
        print("No valid entries found.")
        return

    
    base_time = None
    for ts, qlen in queue_data:
        if qlen == 2048:
            base_time = ts
            break

    if not base_time:
        base_time = queue_data[0][0]  

    prev_wait_ms = 0
    for i in range(1, len(queue_data)):
        t0, q0 = queue_data[i - 1]
        t1, q1 = queue_data[i]
        delta_t = (t1 - t0).total_seconds()
        delta_q = q0 - q1
        tput = delta_q / delta_t if delta_t > 0 else 0
        wait_ms = (q1 / tput) * 1000 if tput > 0 else 0
        if wait_ms > 0:
            prev_wait_ms = wait_ms
        else:
            wait_ms = prev_wait_ms
        rel_time = int((t1 - base_time).total_seconds() * 1000)

        print(f"{q1}\t{rel_time}\t{round(wait_ms, 2)}")
        # print(f"{q1},{round(wait_ms, 2)}")

if __name__ == "__main__":
    parse_log_wait_time("/tmp/iluvatar/logs/worker.log")



# import json
# from datetime import datetime

# def extract_cpu_queue_len_with_relative_time(log_path):
#     results = []

#     # Helper to parse high-precision timestamp
#     def parse_ts(ts_str):
#         return datetime.strptime(ts_str[:26], "%Y-%m-%d %H:%M:%S.%f")

#     with open(log_path, 'r') as f:
#         for line in f:
#             try:
#                 entry = json.loads(line)
#                 fields = entry.get("fields", {})
#                 if fields.get("message") == "current load status":
#                     timestamp_str = entry.get("timestamp")
#                     status_json = fields.get("status")
#                     if status_json:
#                         status = json.loads(status_json)
#                         cpu_queue_len = status.get("cpu_queue_len")
#                         if cpu_queue_len is not None:
#                             timestamp = parse_ts(timestamp_str)
#                             results.append((timestamp, cpu_queue_len))
#             except (json.JSONDecodeError, ValueError):
#                 continue  # skip any malformed lines
# # timestamp":"2025-04-24 16:09:40.713978545
#     # Convert timestamps to relative milliseconds
#     if results:
#         base_time = results[0][0]
#         rel_results = [(0, results[0][1])]
#         for t, qlen in results[1:]:
#             delta_ms = (t - base_time).total_seconds() * 1000
#             rel_results.append((int(delta_ms), qlen))
#         return rel_results
#     return []

# # Example usage
# log_file = "/tmp/iluvatar/logs/worker.log"  # replace with actual file
# ms_tmp = 0  
# for ms, qlen in extract_cpu_queue_len_with_relative_time(log_file):
#     # print(f"{ms} ms -> cpu_queue_len: {qlen}")
#     # if ms > 873806:
#     ms_tmp = ms - ms_tmp
#     print(f"{ms_tmp}\t{qlen}")
#     ms_tmp = ms
    
# print()