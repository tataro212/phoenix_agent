"""
Station 1: The Surveyor (document_parser.py)
Creates a flawless, complete document_blueprint.json from the source PDF using YOLO (DocLayNet), Nougat, and PyMuPDF.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
import cv2
import numpy as np
from document_blueprint import (
    DocumentBlueprint, create_blueprint, create_element, create_page, save_blueprint, validate_blueprint
)

# --- Placeholders for YOLO and Nougat model loading ---
# In a real implementation, replace these with actual model loading and inference code
class DummyYOLOModel:
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # Return dummy results: [{"bbox": (x0, y0, x1, y1), "class_label": "paragraph", "confidence": 0.99}]
        return []

class DummyNougatModel:
    def process(self, image: np.ndarray) -> str:
        # Return dummy text
        return ""

# --- Non-Maximal Suppression (NMS) ---
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # boxes: List of [x0, y0, x1, y1]
    # scores: List of confidence scores
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# --- Main Parsing Function ---
def parse_pdf_to_blueprint(pdf_path: str, yolo_model_path: str = None, nougat_model_path: str = None, output_json: str = "document_blueprint.json") -> DocumentBlueprint:
    """
    Parses a PDF into a DocumentBlueprint using YOLO, Nougat, and PyMuPDF.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    blueprint = create_blueprint(str(pdf_path))
    
    # Load models (replace with real loading code)
    yolo_model = DummyYOLOModel()
    nougat_model = DummyNougatModel()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        width, height = page.rect.width, page.rect.height
        page_obj = create_page(page_num + 1, width, height)
        
        # --- Pass 1: High-Level Layout Segmentation (YOLO) ---
        # Render page to 300 DPI image
        zoom = 300 / 72  # 72 DPI is default
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        
        # YOLO prediction
        yolo_results = yolo_model.predict(img)
        # Group by class_label for NMS
        class_to_boxes = {}
        class_to_scores = {}
        for res in yolo_results:
            label = res["class_label"]
            class_to_boxes.setdefault(label, []).append(res["bbox"])
            class_to_scores.setdefault(label, []).append(res["confidence"])
        filtered_boxes = []
        for label, boxes in class_to_boxes.items():
            scores = class_to_scores[label]
            keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
            for idx in keep:
                filtered_boxes.append({"bbox": boxes[idx], "class_label": label, "confidence": scores[idx]})
        # No parent-child logic in dummy; real code would handle this
        
        # --- Pass 2: Detailed Content & Property Extraction ---
        elements = []
        for idx, box in enumerate(filtered_boxes):
            bbox = box["bbox"]
            label = box["class_label"]
            # Crop image for Nougat
            margin = 5
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(img.shape[1], x1 + margin)
            y1 = min(img.shape[0], y1 + margin)
            cropped_img = img[y0:y1, x0:x1]
            # Nougat OCR
            content = nougat_model.process(cropped_img)
            # PyMuPDF property extraction
            text_instances = page.get_text("dict", clip=(bbox[0], bbox[1], bbox[2], bbox[3]))
            # Heuristics (dummy values for now)
            font_name = "Arial"
            font_size = 12.0
            font_weight = "normal"
            is_italic = False
            text_color = (0.0, 0.0, 0.0)
            alignment = "left"
            indentation = 0.0
            list_level = 0
            list_style = "none"
            # TODO: Implement real heuristics based on text_instances
            # Type refinement
            element_type = label
            element_id = f"p{page_num+1}_e{idx}"
            element = create_element(
                element_id=element_id,
                element_type=element_type,
                bbox=(bbox[0]/img.shape[1], bbox[1]/img.shape[0], bbox[2]/img.shape[1], bbox[3]/img.shape[0]),
                content=content,
                font_name=font_name,
                font_size=font_size,
                font_weight=font_weight,
                is_italic=is_italic,
                text_color=text_color,
                alignment=alignment,
                indentation=indentation,
                list_level=list_level,
                list_style=list_style
            )
            elements.append(element)
        page_obj["elements"] = elements
        # --- Pass 3: Reading Order Determination ---
        # Sort by top (bbox[1]), then left (bbox[0])
        sorted_elements = sorted(elements, key=lambda el: (el["bbox"][1], el["bbox"][0]))
        page_obj["reading_order"] = [el["id"] for el in sorted_elements]
        blueprint["pages"].append(page_obj)
    # Save blueprint
    save_blueprint(blueprint, output_json)
    print(f"Blueprint saved to {output_json}")
    return blueprint

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_parser.py <input.pdf> [output.json]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "document_blueprint.json"
    parse_pdf_to_blueprint(pdf_path, output_json=output_json) 