#
# import time
# from logger import log_result
# # from execute_MES_async_WKT import QoED_test
#
# from ckn_controller.ckn_main import main_ensemble_invoke
#
# async def send_request(req_id, deadline, iar_sec, mode, current_time_sec):
#     start = time.time()
#     start1 = time.perf_counter()
#     try:
#         result = await main_ensemble_invoke(transaction_id=req_id, deadline=deadline)
#         duration = time.time() - start
#         duration1 = time.perf_counter() - start1
#         log_result(mode, req_id, deadline, iar_sec, duration,current_time_sec, result)
#     except Exception as e:
#         duration = time.time() - start
#         log_result(mode, req_id, deadline, iar_sec, duration, current_time_sec,{
#             "model": None,
#             "accuracy": 0.0,
#             "latency": -1,
#             "container_state": "ERROR",
#             "success": False,
#             "error": str(e)
#         })



# import time
# import aiohttp
# from logger import log_result
#
# REMOTE_CKN_URL = "http://127.0.0.1:9000/invoke"
#
# async def call_remote_ckn(req_id: str, deadline: int) -> dict:
#     print("11111111111111")
#     payload = {"transaction_id": req_id, "deadline": deadline}
#     timeout = aiohttp.ClientTimeout(total=600)  # adjust if needed
#     async with aiohttp.ClientSession(timeout=timeout) as session:
#         async with session.post(REMOTE_CKN_URL, json=payload) as resp:
#             resp.raise_for_status()
#             return await resp.json()
#
# async def send_request(req_id, deadline, iar_sec, mode, current_time_sec):
#     start = time.time()
#     start1 = time.perf_counter()
#     try:
#         result = await call_remote_ckn(req_id, deadline)
#
#         duration = time.time() - start
#         duration1 = time.perf_counter() - start1
#         log_result(mode, req_id, deadline, iar_sec, duration, current_time_sec, result)
#
#     except Exception as e:
#         duration = time.time() - start
#         log_result(mode, req_id, deadline, iar_sec, duration, current_time_sec, {
#             "model": None,
#             "accuracy": 0.0,
#             "latency": -1,
#             "container_state": "ERROR",
#             "success": False,
#             "error": str(e)
#         })



import time
import aiohttp
from logger import log_result
import base64
from pathlib import Path
import aiohttp

import os
import random
import base64
from pathlib import Path
import aiohttp

REMOTE_CKN_URL = "http://127.0.0.1:9000/invoke"  # or remote IP

BASE_FOLDER = "/Users/agamage/Desktop/D2I/Codes Original/Mode-S/archive/train.X1"
IMG_EXTS = (".jpg", ".jpeg", ".png")

def pick_random_image(base_folder: str) -> tuple[str, str]:
    """
    Returns: (image_path, selected_folder)
    selected_folder is the ground-truth folder name (WNID).
    """
    folders = [
        f for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f))
    ]
    if not folders:
        raise FileNotFoundError(f"No folders found in {base_folder}")

    selected_folder = random.choice(folders)
    folder_path = os.path.join(base_folder, selected_folder)

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(IMG_EXTS)
    ]
    if not images:
        raise FileNotFoundError(f"No images found in {folder_path}")

    random_image = random.choice(images)
    image_path = os.path.join(folder_path, random_image)
    return image_path, selected_folder

def img_to_b64(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")

async def call_remote_ckn(req_id: str, deadline: int) -> dict:
    # ✅ same selection logic as old main_ensemble_invoke()
    image_path, selected_folder = pick_random_image(BASE_FOLDER)
    image_b64 = img_to_b64(image_path)

    payload = {
        "transaction_id": req_id,
        "deadline": deadline,
        "image_b64": image_b64,
        "selected_folder": selected_folder,  # ground truth
    }

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(REMOTE_CKN_URL, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

async def send_request(req_id, deadline, iar_sec, mode, current_time_sec):
    start = time.time()
    start1 = time.perf_counter()
    try:
        result = await call_remote_ckn(req_id, deadline)

        duration = time.time() - start
        duration1 = time.perf_counter() - start1
        log_result(mode, req_id, deadline, iar_sec, duration, current_time_sec, result)
        print("result: ",result)

    except Exception as e:
        duration = time.time() - start
        log_result(mode, req_id, deadline, iar_sec, duration, current_time_sec, {
            "model": None,
            "accuracy": 0.0,
            "latency": -1,
            "container_state": "ERROR",
            "success": False,
            "error": str(e)
        })