from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import sys
from pathlib import Path

# Add yolov5 path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
sys.path.append(str(ROOT / "yolov5"))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

device = select_device('')
model = DetectMultiBackend('yolov5s.pt', device=device)

app = FastAPI()

@app.post("/count-crops/")
async def count_crops(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()

    pred = model(img_tensor.to(device), augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45)

    crop_count = len(pred[0]) if pred[0] is not None else 0
    return JSONResponse(content={"crop_count": crop_count})
