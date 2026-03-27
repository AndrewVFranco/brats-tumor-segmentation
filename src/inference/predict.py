import numpy as np
import tempfile
import torch
import os
import nibabel as nib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from monai.inferers import sliding_window_inference
from src.training.model import get_model
from src.preprocessing.preprocess import preprocess_array

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
@asynccontextmanager
async def lifespan(app):
    # Run on startup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    model = get_model().to(device)
    app.state.model = model
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    yield
    # Run on shutdown

app = FastAPI(lifespan=lifespan)
@app.post("/segment")
async def segment(request: Request,
                  background_tasks: BackgroundTasks,
                  t1c: UploadFile = File(...),
                  t1n: UploadFile = File(...),
                  t2f: UploadFile = File(...),
                  t2w: UploadFile = File(...)):

    try:
        model = request.app.state.model
        device = request.app.state.device

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp.write(await t1c.read())
            t1c_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp.write(await t1n.read())
            t1n_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp.write(await t2f.read())
            t2f_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp.write(await t2w.read())
            t2w_path = tmp.name

        affine = nib.load(t1c_path)
        modality_arrays = {"t1c": affine.get_fdata(),
                           "t1n": nib.load(t1n_path).get_fdata(),
                           "t2f": nib.load(t2f_path).get_fdata(),
                           "t2w": nib.load(t2w_path).get_fdata()
                           }

        modality_arrays = preprocess_array(modality_arrays)

        image = np.stack((modality_arrays["t1c"],
                         modality_arrays["t1n"],
                         modality_arrays["t2f"],
                         modality_arrays["t2w"]),
                         axis=0
                         )

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        output = sliding_window_inference(image,
                                          roi_size=(128, 128, 128),
                                          sw_batch_size=1,
                                          predictor=model
                                          )

        # Convert to segmentation mask
        seg_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save as NIfTI using the affine from the input
        seg_nifti = nib.Nifti1Image(seg_mask, affine.affine)

        # Save to a temp file
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            nib.save(seg_nifti, tmp.name)
            output_path = tmp.name

        # Temp file cleanup
        for path in [t1c_path, t1n_path, t2f_path, t2w_path]:
            os.unlink(path)

        background_tasks.add_task(os.unlink, output_path)

        # Return the file
        return FileResponse(output_path, media_type="application/octet-stream", filename="segmentation.nii")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}