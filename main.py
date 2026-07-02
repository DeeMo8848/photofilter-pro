"""
PhotoFilter Pro — FastAPI Backend
"""

import os
import io
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from starlette.requests import Headers
import uvicorn
from urllib.parse import quote

from processor import load_and_process, DEFAULT_PARAMS, set_use_gpu, get_gpu_status, HAS_GPU

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
PRESETS_DIR = BASE_DIR / "presets"

# Ensure dirs
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PRESETS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PhotoFilter Pro", version="1.0.0")


# ═══════════════════════════════════════════
#  Static & Template
# ═══════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    template = BASE_DIR / "templates" / "index.html"
    return template.read_text(encoding="utf-8")


# ═══════════════════════════════════════════
#  Upload
# ═══════════════════════════════════════════

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a single image for preview."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'):
        raise HTTPException(400, f"Unsupported format: {ext}")

    img_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{img_id}{ext}"
    content = await file.read()
    save_path.write_bytes(content)

    return {
        "id": img_id,
        "filename": file.filename,
        "ext": ext,
        "size": len(content)
    }


@app.post("/api/upload/batch")
async def upload_batch(files: list[UploadFile] = File(...)):
    """Upload multiple images for batch processing."""
    results = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'):
            continue
        img_id = uuid.uuid4().hex
        save_path = UPLOAD_DIR / f"{img_id}{ext}"
        content = await file.read()
        save_path.write_bytes(content)
        results.append({
            "id": img_id,
            "filename": file.filename,
            "ext": ext,
            "size": len(content)
        })
    return {"files": results}


# ═══════════════════════════════════════════
#  Preview
# ═══════════════════════════════════════════

_preview_cache: dict = {}  # {img_id: jpeg_bytes} — cached for manual save


@app.get("/api/preview/{img_id}")
async def get_preview(img_id: str):
    """Serve the uploaded image (no params → original)."""
    for ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
        path = UPLOAD_DIR / f"{img_id}{ext}"
        if path.exists():
            return FileResponse(path, media_type=f"image/{ext[1:]}")
    raise HTTPException(404, "Image not found")


@app.post("/api/preview/{img_id}")
async def preview_with_params(img_id: str, params_json: str = Form(...)):
    """Process image with given params and return the result.
    Does NOT auto-save to disk — only caches in memory for explicit save."""
    # Find the file
    src_path = None
    for ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
        p = UPLOAD_DIR / f"{img_id}{ext}"
        if p.exists():
            src_path = p
            break
    if not src_path:
        raise HTTPException(404, "Image not found")

    params = json.loads(params_json)

    # Process
    result = load_and_process(str(src_path), params)

    # Cache in memory for explicit save
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=92)
    _preview_cache[img_id] = buf.getvalue()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/api/save/{img_id}")
async def save_image(img_id: str):
    """Save the processed preview or original to outputs/."""
    if img_id in _preview_cache:
        out_path = OUTPUT_DIR / f"{img_id}.jpg"
        out_path.write_bytes(_preview_cache[img_id])
        return {"ok": True, "path": str(out_path)}
    # No processed cache — save the untouched original
    for ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
        src = UPLOAD_DIR / f"{img_id}{ext}"
        if src.exists():
            out_path = OUTPUT_DIR / f"{img_id}{ext}"
            shutil.copy2(str(src), str(out_path))
            return {"ok": True, "path": str(out_path)}
    raise HTTPException(404, "Image not found")


# ═══════════════════════════════════════════
#  Presets
# ═══════════════════════════════════════════

@app.get("/api/presets")
async def list_presets():
    """List all saved presets."""
    presets = []
    for f in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            presets.append({
                "id": f.stem,
                "name": data.get("name", f.stem),
                "params": data.get("params", {}),
                "created": data.get("created", ""),
            })
        except Exception:
            continue
    return {"presets": presets}


@app.post("/api/presets")
async def save_preset(name: str = Form(...), params_json: str = Form(...)):
    """Save current parameters as a preset."""
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-")[:50]
    preset_id = uuid.uuid4().hex[:8]

    preset_data = {
        "name": name,
        "params": json.loads(params_json),
        "created": datetime.now().isoformat(),
        "id": preset_id
    }

    (PRESETS_DIR / f"{preset_id}.json").write_text(
        json.dumps(preset_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {"id": preset_id, "name": name}


@app.get("/api/presets/{preset_id}")
async def get_preset(preset_id: str):
    """Get a single preset by ID."""
    path = PRESETS_DIR / f"{preset_id}.json"
    if not path.exists():
        raise HTTPException(404, "Preset not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/api/presets/{preset_id}")
async def delete_preset(preset_id: str):
    """Delete a preset."""
    path = PRESETS_DIR / f"{preset_id}.json"
    if not path.exists():
        raise HTTPException(404, "Preset not found")
    path.unlink()
    return {"ok": True}


@app.post("/api/presets/export")
async def export_preset(params_json: str = Form(...), name: str = Form("preset")):
    """Export a preset as downloadable JSON."""
    preset_data = {
        "name": name,
        "params": json.loads(params_json),
        "exported": datetime.now().isoformat()
    }
    content = json.dumps(preset_data, ensure_ascii=False, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name + '.json')}"}
    )


# ═══════════════════════════════════════════
#  GPU Control
# ═══════════════════════════════════════════

@app.get("/api/status")
async def system_status():
    """System status: GPU availability, CPU cores, etc."""
    return {
        "gpu": get_gpu_status(),
        "cpu_cores": multiprocessing.cpu_count(),
        "version": "1.1.0"
    }


@app.post("/api/gpu/toggle")
async def toggle_gpu(enabled: str = Form("false")):
    """Enable/disable GPU acceleration."""
    enable = enabled.lower() in ("true", "1", "yes")
    try:
        set_use_gpu(enable)
        return {"ok": True, "gpu": get_gpu_status()}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ═══════════════════════════════════════════
#  Batch Processing (parallel)
# ═══════════════════════════════════════════

# Lock for GPU-based processing (CuPy can't be shared across processes)
_batch_lock = None


def _process_one_file(args: tuple) -> tuple:
    """Worker function for parallel batch processing."""
    src_path, out_path, params_json = args
    params = json.loads(params_json)
    try:
        result = load_and_process(src_path, params)
        result.save(out_path, "JPEG", quality=92)
        return (os.path.basename(src_path), True, None)
    except Exception as e:
        return (os.path.basename(src_path), False, str(e))


@app.post("/api/batch")
async def batch_process(params_json: str = Form(...), file_ids: str = Form(...),
                        workers: str = Form("auto")):
    """
    Batch process multiple images with the same params.
    file_ids: comma-separated list of image IDs
    workers: number of parallel workers ("auto" = CPU count, or a number)
    Returns: zip file with all processed images
    """
    ids = [fid.strip() for fid in file_ids.split(",") if fid.strip()]
    params = json.loads(params_json)

    if not ids:
        raise HTTPException(400, "No files specified")

    # Resolve source paths
    tasks = []
    batch_id = uuid.uuid4().hex
    batch_dir = OUTPUT_DIR / f"batch_{batch_id}"
    batch_dir.mkdir(exist_ok=True)

    for img_id in ids:
        src_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            p = UPLOAD_DIR / f"{img_id}{ext}"
            if p.exists():
                src_path = str(p)
                break
        if src_path:
            out_path = str(batch_dir / f"{img_id}.jpg")
            tasks.append((src_path, out_path, params_json))

    if not tasks:
        raise HTTPException(400, "No valid source files found")

    # Determine worker count
    if workers == "auto":
        n_workers = min(multiprocessing.cpu_count(), len(tasks), 8)
    else:
        n_workers = max(1, min(int(workers), len(tasks)))

    # Parallel processing
    processed = []
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_file, t): t for t in tasks}
        for future in as_completed(futures):
            fname, ok, err = future.result()
            if ok:
                processed.append(fname)
            else:
                errors.append({"file": fname, "error": err})

    # Create zip
    # (processing complete, files saved to batch_dir)
    return {
        "processed": len(processed),
        "total": len(tasks),
        "errors": errors,
        "workers_used": n_workers,
        "batch_id": batch_id,
        "output_dir": str(batch_dir)
    }


# ═══════════════════════════════════════════
#  Default params
# ═══════════════════════════════════════════

@app.get("/api/defaults")
async def get_defaults():
    """Return default parameters."""
    return DEFAULT_PARAMS


# ═══════════════════════════════════════════
#  Startup
# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════

@app.post("/api/open_folder")
async def open_folder(path: str = Form(...)):
    """Open folder in Windows Explorer."""
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    os.startfile(str(p))
    return {"ok": True}


if __name__ == "__main__":
    # Clean up stale uploads from previous runs
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
    uvicorn.run(app, host="0.0.0.0", port=8899)
