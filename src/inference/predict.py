import numpy as np
import base64
import tempfile
import torch
import os
import nibabel as nib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from monai.inferers import sliding_window_inference
from src.training.model import get_model
from src.preprocessing.preprocess import preprocess_array
from src.training.transforms import get_inference_transforms

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
app.mount("/static", StaticFiles(directory="src/interface"), name="static")

@app.get("/")
async def root():
    with open("src/interface/index.html") as f:
        return HTMLResponse(f.read())

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

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(await t1c.read())
            t1c_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(await t1n.read())
            t1n_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(await t2f.read())
            t2f_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
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

        tensors = {"image": image}
        tensors = get_inference_transforms()(tensors)
        image = tensors["image"]

        image = image.float().unsqueeze(0).to(device)

        output = sliding_window_inference(image,
                                          roi_size=(128, 128, 128),
                                          sw_batch_size=1,
                                          predictor=model
                                          )

        # Convert to segmentation mask
        seg_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save as NIfTI using the affine from the input
        seg_nifti = nib.Nifti1Image(seg_mask, np.eye(4))

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            nib.save(seg_nifti, tmp.name)
            output_path = tmp.name

        # Save preprocessed T1c as NIfTI with identity affine for display
        display_t1c = modality_arrays["t1c"].astype(np.float32)
        # Push the exact 0 background below the darkest Z-scored brain tissue
        display_t1c[display_t1c == 0] = display_t1c.min() - 1

        # Save preprocessed T1c as NIfTI with identity affine for display
        t1c_nifti = nib.Nifti1Image(display_t1c, np.eye(4))

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            nib.save(t1c_nifti, tmp.name)
            t1c_processed_path = tmp.name

        # Cleanup uploaded temp files
        for path in [t1c_path, t1n_path, t2f_path, t2w_path]:
            os.unlink(path)

        # Read both output files as base64
        with open(output_path, 'rb') as f:
            seg_b64 = base64.b64encode(f.read()).decode()

        with open(t1c_processed_path, 'rb') as f:
            t1c_b64 = base64.b64encode(f.read()).decode()

        # Schedule cleanup of output temp files
        background_tasks.add_task(os.unlink, output_path)
        background_tasks.add_task(os.unlink, t1c_processed_path)

        # Return both files as base64 JSON
        return {"segmentation": seg_b64, "t1c": t1c_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}