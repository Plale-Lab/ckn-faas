import pandas as pd

CSV_PATH = "results/frugalml_E0.120.csv"

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))

if "success" in df.columns:
    print("Successful requests:", df["success"].sum())
    print("Failed requests:", (~df["success"].astype(bool)).sum())

if "accuracy" in df.columns:
    print("Overall accuracy:", df["accuracy"].mean())

if "frugalml_deferred" in df.columns:
    print("Deferral rate:", df["frugalml_deferred"].astype(bool).mean())

if "RespTime" in df.columns:
    print("Average latency (ms):", df["RespTime"].mean())
    print("Median latency (ms):", df["RespTime"].median())
    print("p95 latency (ms):", df["RespTime"].quantile(0.95))
    print("p99 latency (ms):", df["RespTime"].quantile(0.99))

if "number_of_model_calls" in df.columns:
    print(
        "Average model calls/request:",
        df["number_of_model_calls"].mean(),
    )

if "deadline_met" in df.columns:
    print(
        "Deadline miss rate:",
        1.0 - df["deadline_met"].astype(bool).mean(),
    )

print("\nPer-IAR summary:")

summary = (
    df.groupby("IAR")
    .agg(
        requests=("ID", "count"),
        accuracy=("accuracy", "mean"),
        avg_latency_ms=("RespTime", "mean"),
        p95_latency_ms=("RespTime", lambda x: x.quantile(0.95)),
        p99_latency_ms=("RespTime", lambda x: x.quantile(0.99)),
        deferral_rate=("frugalml_deferred", "mean"),
        avg_model_calls=("number_of_model_calls", "mean"),
    )
    .reset_index()
    .sort_values("IAR", ascending=False)
)

print(summary.to_string(index=False))