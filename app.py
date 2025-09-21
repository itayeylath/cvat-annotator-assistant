import os
import io
import httpx
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from shapely.geometry import Polygon
from typing import Optional

# --- Config via env ---
CVAT_BASE_URL = os.getenv("CVAT_BASE_URL", "http://localhost:8080")
CVAT_TOKEN = os.getenv("CVAT_TOKEN", "")  # create in CVAT: Account → Tokens

app = FastAPI(title="CVAT Section Assistant (OpenCV)")

# ==== Schemas ====

class Params(BaseModel):
    image_url: str
    job_id: Optional[int] = None
    label_id: Optional[int] = None
    frame: int = 0

    # Tuning
    min_area: int = 4000
    canny1: int = 60
    canny2: int = 180
    approx_epsilon: float = 0.02
    rect_axis_align_tolerance: float = 10.0
    min_aspect: float = 0.15
    max_aspect: float = 6.0

# ==== Helpers ====

async def fetch_image(url: str) -> np.ndarray:
    # Handle local file URLs
    if url.startswith('file://'):
        file_path = url[7:]  # Remove 'file://' prefix
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(400, f"Failed to load local image: {file_path}")
        return img
    
    # Handle HTTP/HTTPS URLs
    headers = {}
    if CVAT_TOKEN and 'localhost:8080' in url:
        headers['Authorization'] = f'Token {CVAT_TOKEN}'
    
    print(f"DEBUG: Fetching image from URL: {url}")  # Debug logging
    print(f"DEBUG: Headers: {headers}")  # Debug logging
    
    async with httpx.AsyncClient(timeout=60) as cl:
        r = await cl.get(url, headers=headers)
        print(f"DEBUG: Response status: {r.status_code}")  # Debug logging
        print(f"DEBUG: Response text: {r.text[:200]}")  # Debug logging
        # Don't use raise_for_status() for CVAT data endpoints as they don't support HEAD requests
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Failed to fetch image: {r.text}")
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Failed to decode image")
        return img

def contour_to_quad(cnt: np.ndarray, epsilon_ratio: float) -> Optional[np.ndarray]:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(-1, 2)
    return None

def quad_area(quad: np.ndarray) -> float:
    return float(abs(cv2.contourArea(quad.reshape(-1,1,2))))

def quad_aspect(quad: np.ndarray) -> float:
    rect = cv2.minAreaRect(quad.astype(np.float32))
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return 9999.0
    major = max(w, h)
    minor = min(w, h)
    return float(major / (minor + 1e-6))

def is_axis_aligned_quad(quad: np.ndarray, tol_deg: float) -> bool:
    # Use the rotation of the min-area rect as a proxy for axis alignment
    rect = cv2.minAreaRect(quad.astype(np.float32))
    rot = abs(rect[2])
    rot = min(rot, abs(90 - rot))
    return rot <= tol_deg

def order_quad_points(quad: np.ndarray) -> np.ndarray:
    # Return TL, TR, BR, BL
    pts = quad.astype(float)
    s = pts.sum(axis=1)        # TL has min sum, BR has max sum
    d = np.diff(pts, axis=1)   # TR has min diff, BL has max diff
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=float)

def to_cvat_polygon(quad: np.ndarray, label_id: int, frame: int) -> Optional[dict]:
    ordered = order_quad_points(quad)
    polygon = Polygon(ordered)
    if not polygon.is_valid or polygon.area <= 0:
        return None
    flat = [float(v) for v in ordered.reshape(-1)]
    return {
        "type": "polygon",
        "label_id": int(label_id),
        "points": flat,
        "frame": int(frame),
        "group": 0,
        "z_order": 0,
        "attributes": [],
    }

def find_rectangles(img: np.ndarray, p: Params) -> list[np.ndarray]:
    """Return list of 4-point quads (np.ndarray shape (4,2))."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, p.canny1, p.canny2)

    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quads = []
    for cnt in contours:
        quad = contour_to_quad(cnt, p.approx_epsilon)
        if quad is None:
            continue
        area = quad_area(quad)
        if area < p.min_area:
            continue
        if not is_axis_aligned_quad(quad, p.rect_axis_align_tolerance):
            continue
        asp = quad_aspect(quad)
        if asp < p.min_aspect or asp > p.max_aspect:
            continue
        quads.append(quad)

    # Deduplicate overlapping quads by keeping larger ones
    kept = []
    for q in sorted(quads, key=lambda q: -quad_area(q)):
        poly_q = Polygon(q)
        if any(poly_q.intersection(Polygon(qq)).area / (poly_q.area + 1e-6) > 0.8 for qq in kept):
            continue
        kept.append(q)
    return kept
# ==== Endpoints ====

@app.post("/auto_sections")
async def auto_sections(p: Params):
    print(f"***** itay test")
    if not CVAT_TOKEN:
        raise HTTPException(400, "Missing CVAT_TOKEN env")
    if p.job_id is None or p.label_id is None:
        raise HTTPException(400, "job_id and label_id are required")

    img = await fetch_image(p.image_url)
    quads = find_rectangles(img, p)
    print(f"***** quads: {quads}")
    shapes = []
    for q in quads:
        poly = to_cvat_polygon(q, p.label_id, p.frame)
        if poly:
            shapes.append(poly)

    async with httpx.AsyncClient(timeout=60) as cl:
        url = f"{CVAT_BASE_URL}/api/jobs/{p.job_id}/annotations?action=create"
        headers = {
            "Authorization": f"Token {CVAT_TOKEN}",
            "Content-Type": "application/json",
        }
        r = await cl.patch(url, headers=headers, json={"shapes": shapes})
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"CVAT PATCH failed: {r.text}")
    return {"patched": len(shapes)}
