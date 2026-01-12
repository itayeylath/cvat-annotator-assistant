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
CVAT_BASE_URL = os.getenv("CVAT_BASE_URL", "http://cvat-prod.shopperai.ai/")  # TODO: fix
CVAT_TOKEN = os.getenv("CVAT_TOKEN", "9208f2c8b44dd6e7692a3b1f6c2e07c11280a265") # TODO: fix

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

class ClearParams(BaseModel):
    job_id: int

class PolygonItem(BaseModel):
    label_id: int
    frame: int
    points: list[float]
    group: int = 0

class PushPolygonsRequest(BaseModel):
    job_id: int
    polygons: list[PolygonItem]

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
    # quads = refal.function()
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

@app.post("/push_polygons_to_cvat")
async def push_polygons_to_cvat(request: PushPolygonsRequest):
    """
    Push pre-computed polygons directly to CVAT.
    
    Request body format:
    {
        "job_id": 123,
        "polygons": [
            {
                "label_id": 1,
                "frame": 0,
                "points": [x1, y1, x2, y2, ...],
                "group": 0
            }
        ]
    }
    """
    if not CVAT_TOKEN:
        raise HTTPException(400, "Missing CVAT_TOKEN env")
    
    # Convert to CVAT shapes format
    shapes = []
    for polygon in request.polygons:
        shape = {
            "type": "polygon",
            "occluded": False,
            "z_order": 0,
            "rotation": 0.0,
            "points": polygon.points,
            "label_id": polygon.label_id,
            "group": polygon.group,
            "frame": polygon.frame,
            "attributes": []
        }
        shapes.append(shape)
    
    # Push to CVAT
    async with httpx.AsyncClient(timeout=60) as cl:
        url = f"{CVAT_BASE_URL}/api/jobs/{request.job_id}/annotations?action=create"
        headers = {
            "Authorization": f"Token {CVAT_TOKEN}",
            "Content-Type": "application/json",
        }
        
        payload = {"shapes": shapes}
        r = await cl.patch(url, headers=headers, json=payload)
        
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"CVAT PATCH failed: {r.text}")
    
    return {
        "success": True,
        "job_id": request.job_id,
        "polygons_created": len(shapes),
        "message": f"Successfully created {len(shapes)} polygons in CVAT job {request.job_id}"
    }

@app.post("/clear_sections")
async def clear_sections(p: ClearParams):
    """Remove all annotations from a CVAT job."""
    if not CVAT_TOKEN:
        raise HTTPException(400, "Missing CVAT_TOKEN env")

    async with httpx.AsyncClient(timeout=60) as cl:
        url = f"{CVAT_BASE_URL}/api/jobs/{p.job_id}/annotations"
        headers = {
            "Authorization": f"Token {CVAT_TOKEN}",
            "Content-Type": "application/json",
        }
        
        # DELETE request to remove all annotations
        r = await cl.delete(url, headers=headers)
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"CVAT DELETE failed: {r.text}")
    
    return {"message": "All annotations cleared from job", "job_id": p.job_id}

@app.get("/get_polygons")
async def get_polygons(
    job_id: int = Query(..., description="CVAT job ID"),
    label_id: Optional[int] = Query(None, description="Optional label ID to filter polygons")
):
    """
    Fetch polygon annotations (label coordinates) from a CVAT job.
    Returns a list of polygons with coordinates and metadata.
    """
    if not CVAT_TOKEN:
        raise HTTPException(400, "Missing CVAT_TOKEN env")

    async with httpx.AsyncClient(timeout=60) as cl:
        url = f"{CVAT_BASE_URL}/api/jobs/{job_id}/annotations"
        headers = {
            "Authorization": f"Token {CVAT_TOKEN}",
            "Content-Type": "application/json",
        }

        r = await cl.get(url, headers=headers)
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"Failed to fetch polygons: {r.text}")
        
        data = r.json()
        shapes = data.get("shapes", [])
        
        # Filter only polygons (not rectangles, points, etc.)
        polygons = [s for s in shapes if s.get("type") == "polygon"]

        if label_id is not None:
            polygons = [s for s in polygons if s.get("label_id") == label_id]

        # Simplify the output
        simplified = [
            {
                "label_id": p["label_id"],
                "frame": p["frame"],
                "points": p["points"],  # flat list [x1, y1, x2, y2, ...]
                "group": p.get("group", 0),
            }
            for p in polygons
        ]
        return {"job_id": job_id, "count": len(simplified), "polygons": simplified}

@app.get("/get_sections")
async def get_sections(
    job_id: int = Query(..., description="CVAT job ID"),
    label_id: Optional[int] = Query(None, description="Optional label ID to filter polygons"),
    category_name: str = Query("not_empty", description="Category name to use for sections")
):
    """
    Fetch polygon annotations from CVAT job and return them in sections_0.json format.
    
    Returns an array of section objects, each containing:
    - polygon: Array of [x, y] coordinate pairs (closed polygon, first point = last point)
    - category_name: Category name for the section (default: "not_empty")
    
    Format matches sections_0.json structure used by the product classification pipeline.
    """
    if not CVAT_TOKEN:
        raise HTTPException(400, "Missing CVAT_TOKEN env")

    async with httpx.AsyncClient(timeout=60) as cl:
        url = f"{CVAT_BASE_URL}/api/jobs/{job_id}/annotations"
        headers = {
            "Authorization": f"Token {CVAT_TOKEN}",
            "Content-Type": "application/json",
        }

        r = await cl.get(url, headers=headers)
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"Failed to fetch annotations: {r.text}")
        
        data = r.json()
        shapes = data.get("shapes", [])
        
        # Filter only polygons (not rectangles, points, etc.)
        polygons = [s for s in shapes if s.get("type") == "polygon"]
        
        if label_id is not None:
            polygons = [s for s in polygons if s.get("label_id") == label_id]

        # Convert to sections_0.json format
        sections = []
        for p in polygons:
            points = p.get("points", [])  # Flat list: [x1, y1, x2, y2, ...]
            
            # Convert flat list to array of [x, y] pairs
            polygon_coords = []
            if len(points) >= 6:  # At least 3 points (6 coordinates)
                for i in range(0, len(points), 2):
                    if i + 1 < len(points):
                        polygon_coords.append([float(points[i]), float(points[i + 1])])
            
            # Close the polygon if needed (first point should equal last point)
            if len(polygon_coords) >= 3:
                # Check if polygon is already closed
                first_point = polygon_coords[0]
                last_point = polygon_coords[-1]
                if first_point != last_point:
                    polygon_coords.append(first_point.copy())  # Close the polygon
                
                # Extract category_name from attributes if available
                section_category = category_name  # Default
                attributes = p.get("attributes", [])
                for attr in attributes:
                    if isinstance(attr, dict):
                        attr_name = str(attr.get('name', '')).lower()
                        if 'category' in attr_name or 'section' in attr_name:
                            section_category = attr.get('value', category_name)
                
                sections.append({
                    "polygon": polygon_coords,
                    "category_name": section_category
                })
        
        return sections

@app.get("/get_sections_for_catalog")
async def get_sections_for_catalog(
    job_id: int = Query(..., description="CVAT job ID"),
    label_id: Optional[int] = Query(None, description="Optional label ID to filter annotations")
):
    """
    Fetch annotations from CVAT job and return them directly in ground truth JSON format.
    
    Returns a list of ground truth entries, each containing:
    - detection_idx: Index/order of the annotation (0, 1, 2, ...)
    - product_id: Product name extracted from attributes (required)
    - section_label: Section label if available in attributes (optional)
    
    Only entries with a valid product_id are included in the response.
    Entries without product_id are skipped (user needs to fill them manually).
    
    The response is ready to use as ground truth JSON file for product classification.
    """
    if not CVAT_TOKEN:
        raise HTTPException(400, "Missing CVAT_TOKEN env")

    async with httpx.AsyncClient(timeout=60) as cl:
        url = f"{CVAT_BASE_URL}/api/jobs/{job_id}/annotations"
        headers = {
            "Authorization": f"Token {CVAT_TOKEN}",
            "Content-Type": "application/json",
        }

        r = await cl.get(url, headers=headers)
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"Failed to fetch annotations: {r.text}")
        
        data = r.json()
        shapes = data.get("shapes", [])
        
        # Filter only polygons (not rectangles, points, etc.)
        polygons = [s for s in shapes if s.get("type") == "polygon"]
        print(f"***** polygons: {polygons}")
        if label_id is not None:
            polygons = [s for s in polygons if s.get("label_id") == label_id]

        # Extract product names and section labels from attributes
        ground_truth_data = []
        for idx, p in enumerate(polygons):
            attributes = p.get("attributes", [])
            
            # Extract product name from attributes
            product_id = None
            section_label = None
            
            for attr in attributes:
                # Look for product name in attributes
                # Pattern 1: {'spec_id': X, 'value': 'ProductName'}
                if 'value' in attr and attr['value']:
                    value = attr['value']
                    # Skip boolean/numeric values that aren't product names
                    if isinstance(value, str) and value.lower() not in ['true', 'false', '0', '1']:
                        # Check if it looks like a product name (not a section/category)
                        if product_id is None and not any(keyword in value.lower() for keyword in ['section', 'category', 'empty']):
                            product_id = value
                
                # Pattern 2: Look for section/category in attributes
                attr_name = str(attr.get('name', '')).lower()
                if 'section' in attr_name or 'category' in attr_name:
                    section_label = attr.get('value')
            
            # If no product name found, try to get it from label name or other fields
            if product_id is None:
                # You might want to add logic here to extract from label_id mapping
                # For now, we'll leave it as None and let the user fill it in
                pass
            
            # Create entry in ground truth format (only required fields)
            entry = {
                "detection_idx": idx,  # Use annotation order as detection index
                "product_id": product_id,  # Will be None if not found in attributes
            }
            
            # Add section_label only if found
            if section_label:
                entry["section_label"] = section_label
            
            # Only include entries that have a product_id
            # Skip entries without product_id (user needs to fill them manually)
            if product_id is not None:
                ground_truth_data.append(entry)
        
        # Return directly in ground truth JSON format (array of objects)
        return ground_truth_data
