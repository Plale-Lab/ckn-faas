import asyncio
from generator import generate_workload

if __name__ == "__main__":
    # Choose "short", "medium", or "long"
    # asyncio.run(generate_workload("short", duration_sec=600 , mode="azure_trace"))
    # asyncio.run(generate_workload("dynamic", duration_sec=100, mode="dynamic_iar"))
    # asyncio.run(generate_workload("short", duration_sec=0.02, mode="vary_iar"))
    asyncio.run(generate_workload("short", duration_sec=100, mode="vary_deadline"))

    # vary_deadline