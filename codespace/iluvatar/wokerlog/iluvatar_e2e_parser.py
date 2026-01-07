#!/usr/bin/env python3
import os, sys, json, gzip, re
from pathlib import Path
from datetime import datetime
import pandas as pd

def open_maybe_gzip(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if str(path).endswith(".gz") \
           else open(path, "r", encoding="utf-8", errors="ignore")

def to_float(x):
    try: return float(x)
    except: return None

def parse_ts(s):
    if not s: return None
    s = s.strip()
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{(frac+'000000')[:6]}"
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except:
        try: return datetime.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S")
        except: return None

def parse_log(lines):
    rows = {}
    for raw in lines:
        line = raw.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except:
            m = re.search(r'(\{.*\})', line)
            if not m: continue
            try: obj = json.loads(m.group(1))
            except: continue

        fields = obj.get("fields", {}) or {}
        span = obj.get("span", {}) or {}
        target = obj.get("target", "") or ""
        ts = obj.get("timestamp")

        tid = fields.get("tid") or span.get("tid")
        if not tid: continue
        r = rows.setdefault(tid, {"tid": tid})

        # model
        if "fqdn" in fields: r["model_fqdn"] = fields["fqdn"]
        if "fqdn" in span:   r["model_fqdn"] = span["fqdn"]

        # queue markers
        if "insert_time" in fields and "remove_time" in fields:
            r["insert_ts"] = fields["insert_time"]
            r["start_exec_ts"] = fields["remove_time"]

        # http timings (optional)
        if "http_client" in target and fields.get("message") == "close":
            r["http_idle"] = fields.get("time.idle")
            r["http_busy"] = fields.get("time.busy")

        # completion
        if "e2etime" in fields:
            r["e2e_s"] = to_float(fields["e2etime"])   # seconds
            r["compute"] = fields.get("compute")
            if ts: r["end_ts"] = ts

    return list(rows.values())

def model_summary(df):
    g = df.groupby("model_fqdn")
    return pd.DataFrame({
        "model": g["model_fqdn"].first(),
        "mean_e2e_s": g["e2e_s"].mean(),
        "min_e2e_s":  g["e2e_s"].min(),
        "max_e2e_s":  g["e2e_s"].max(),
        "mean_total_worker_time_s": g["total_worker_time_s"].mean(),
        "min_total_worker_time_s":  g["total_worker_time_s"].min(),
        "max_total_worker_time_s":  g["total_worker_time_s"].max(),
        "mean_queue_wait_s": g["queue_wait_s"].mean()
    }).sort_values(["mean_e2e_s","model"])

def main():
    script_dir = Path(__file__).parent
    logs = list(script_dir.glob("*.log")) + list(script_dir.glob("*.log.gz"))
    if not logs:
        print("No .log or .log.gz found here.")
        input("Press Enter to exit..."); sys.exit(1)
    log_path = logs[0]
    print(f"Using log: {log_path.name}")

    with open_maybe_gzip(log_path) as f:
        rows = parse_log(f)
    if not rows:
        print("No invocations found.")
        input("Press Enter to exit..."); sys.exit(1)

    df = pd.DataFrame(rows)
    df = df[(df["model_fqdn"].notna()) & (df["e2e_s"].notna())].copy()
    if df.empty:
        print("No complete rows with model_fqdn & e2e_s.")
        input("Press Enter to exit..."); sys.exit(2)

    # timestamps -> durations
    for c in ("insert_ts","start_exec_ts","end_ts"):
        df[c+"_dt"] = df[c].apply(parse_ts) if c in df.columns else None
    df["queue_wait_s"]       = (df["start_exec_ts_dt"] - df["insert_ts_dt"]).dt.total_seconds()
    df["e2e_from_ts_s"]      = (df["end_ts_dt"] - df["start_exec_ts_dt"]).dt.total_seconds()
    df["total_worker_time_s"]= (df["end_ts_dt"] - df["insert_ts_dt"]).dt.total_seconds()

    # write per-invocation
    inv_cols = ["tid","model_fqdn","compute","e2e_s","e2e_from_ts_s",
                "queue_wait_s","total_worker_time_s","insert_ts","start_exec_ts","end_ts",
                "http_idle","http_busy"]
    (script_dir/"invocations_e2e.csv").write_text(
        df[inv_cols].to_csv(index=False)
    )

    # model-wise summary (both metrics)
    ms = model_summary(df)
    (script_dir/"e2e_by_model.csv").write_text(ms.to_csv(index=False))

    # ---- PRINT BOTH WAYS ----
    print("\n=== Model-wise E2E  ===")
    print(ms[["model","mean_e2e_s","min_e2e_s","max_e2e_s"]]
          .to_string(index=False, float_format="%.6f"))

    print("\n=== Model-wise queue wait Time ===")
    print(ms[["model", "mean_queue_wait_s"]]
          .to_string(index=False, float_format="%.6f"))

    print("\nNote: total_worker_time_s = queue_wait_s + e2e_s ; they match only when mean_queue_wait_s ≈ 0.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()
