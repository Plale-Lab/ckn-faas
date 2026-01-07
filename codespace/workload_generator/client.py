
import time
from logger import log_result
# from execute_MES_async_WKT import QoED_test

from code_for_testing.ckn_main import main_ensemble_invoke

async def send_request(req_id, deadline, iar_sec, mode, current_time_sec):
    start = time.time()
    start1 = time.perf_counter()
    try:
        result = await main_ensemble_invoke(transaction_id=req_id, deadline=deadline)
        duration = time.time() - start
        duration1 = time.perf_counter() - start1
        log_result(mode, req_id, deadline, iar_sec, duration,current_time_sec, result)
    except Exception as e:
        duration = time.time() - start
        log_result(mode, req_id, deadline, iar_sec, duration, current_time_sec,{
            "model": None,
            "accuracy": 0.0,
            "latency": -1,
            "container_state": "ERROR",
            "success": False,
            "error": str(e)
        })
