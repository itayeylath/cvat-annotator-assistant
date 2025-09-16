# cvat-annotator-assistant

A prototype of a labeling assistant locally on your Mac first, then port it to your work environment if it’s successful. 

"keep it simple—run your FastAPI server locally (no Docker), call SAM, return polygons for label_id=section, and POST them into your CVAT job using the annotations API."-
1. Set Up CVAT Locally
2. Create the Adapter (FastAPI server running OpenCV ) + no need for Docker 
3. Test Locally (no CVAT write yet)  
 4.Push (POST) Polygons Directly into CVAT (so they appear like hand-drawn) 

Backend
 * SAM also an option 
* FastAPI that does everything (download → detect → segment → polygon → POST to CVAT).
* 👉 The assistant doesn’t need to guess from a class list of products.
👉 The assistant only needs to auto-draw polygons for “sections” in the image (like planogram regions).   
* CVAT ML backend - Your “sections” assistant can be exposed as an ML backend. Then annotators click Run, and your service returns polygon shapes for the single label (e.g., section). 
*Option 3: Train a custom detector (later)
If you have lots of shelf images annotated with sections, train a YOLO-Seg / Mask R-CNN model.
Input: image → output polygons for “section”.
Integrate as CVAT ML backend.
This is overkill for POC, but the scalable solution if you need precision and generalization.