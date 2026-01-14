from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from codespace.ckn_controller.ckn_main import main_ensemble_invoke

app = FastAPI()

class InvokeReq(BaseModel):
    transaction_id: str
    deadline: int
    image_b64: str
    selected_folder: str | None = None

@app.post("/invoke")
async def invoke(req: InvokeReq):
    return await main_ensemble_invoke(
        transaction_id=req.transaction_id,
        deadline=req.deadline,
        image_b64=req.image_b64,
        selected_folder=req.selected_folder
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
