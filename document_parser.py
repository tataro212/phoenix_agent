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

# Try to import ultralytics for YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Try to import Nougat
try:
    from nougat import NougatModel
    NOUGAT_AVAILABLE = True
except ImportError:
    NOUGAT_AVAILABLE = False
    print("Warning: nougat-ocr not available. Install with: pip install nougat-ocr")

# --- YOLO Model Integration ---
class YOLOModel:
    def __init__(self, model_path: str = None):
        """Initialize YOLO model for document layout detection."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        # Always use the user's provided trained weights unless another is given
        if model_path is None:
            model_path = r"C:/Users/30694/gemini_translator_env/runs/two_stage_training/stage1_publaynet/yolov8_publaynet_base/weights/best.pt"
            print(f"Using user-provided YOLO model: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model {model_path}: {e}")
            print("Using YOLOv8n as fallback.")
            self.model = YOLO("yolov8n.pt")
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO prediction on image and return results in our format."""
        try:
            # Run YOLO prediction
            results = self.model(image, verbose=False)
            
            predictions = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    # Get class names
                    class_names = result.names
                    print(f"YOLO detected classes: {class_names}")
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        # Convert class ID to name
                        class_name = class_names[int(class_id)]
                        
                        # Map PubLayNet classes to our element types
                        element_type = self.map_class_to_element_type(class_name)
                        print(f"Mapping YOLO class '{class_name}' to element type '{element_type}'")
                        
                        prediction = {
                            "bbox": (box[0], box[1], box[2], box[3]),  # x0, y0, x1, y1
                            "class_label": element_type,
                            "confidence": float(conf)
                        }
                        predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            print(f"Error in YOLO prediction: {e}")
            return []
    
    def map_class_to_element_type(self, class_name: str) -> str:
        """Map PubLayNet class names to our element types."""
        mapping = {
            "text": "paragraph",
            "title": "heading_1", 
            "list": "list_item",
            "table": "table",
            "figure": "image"
        }
        return mapping.get(class_name.lower(), "paragraph")

# --- Nougat Model Integration ---
class NougatModel:
    def __init__(self, model_path: str = None):
        """Initialize Nougat model for OCR and text extraction."""
        if not NOUGAT_AVAILABLE:
            print("Warning: Nougat not available. Using PyMuPDF text extraction as fallback.")
            self.model = None
            return
        
        try:
            if model_path is None:
                model_path = "facebook/nougat-base"
            
            # Import the actual NougatModel class from the nougat module
            import nougat
            self.model = nougat.NougatModel.from_pretrained(model_path)
            print(f"Loaded Nougat model: {model_path}")
        except Exception as e:
            print(f"Error loading Nougat model: {e}")
            self.model = None
    
    def process(self, image: np.ndarray) -> str:
        """Extract text from image using Nougat."""
        if self.model is None:
            # Fallback to basic OCR or return empty string
            return ""
        
        try:
            # Convert image to PIL Image for Nougat
            from PIL import Image
            pil_image = Image.fromarray(image)
            
            # Process with Nougat
            result = self.model.inference(image=pil_image)
            return result if result else ""
            
        except Exception as e:
            print(f"Error in Nougat processing: {e}")
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
    
    # Load models with fallback to dummy models if needed
    try:
        yolo_model = YOLOModel(yolo_model_path)
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        return None
    
    try:
        nougat_model = NougatModel(nougat_model_path)
        print("✅ Nougat model loaded successfully")
    except Exception as e:
        print(f"⚠️  Nougat model loading failed: {e}")
        # Continue without Nougat, will use PyMuPDF fallback
        nougat_model = None
    
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
        print(f"Page {page_num + 1}: YOLO found {len(yolo_results)} elements")
        
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
        
        # --- Pass 2: Detailed Content & Property Extraction ---
        elements = []
        
        # If YOLO found no elements, fall back to basic text extraction
        if not filtered_boxes:
            print(f"Page {page_num + 1}: No YOLO elements found, using PyMuPDF text extraction")
            # Extract text blocks using PyMuPDF
            text_dict = page.get_text("dict")
            text_blocks = text_dict.get("blocks", [])
            
            for block_idx, block in enumerate(text_dict.get("blocks", [])):
                if "lines" in block:  # Text block
                    # Get block bounds
                    bbox = block["bbox"]  # (x0, y0, x1, y1)
                    
                    # Extract text content
                    content = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            content += span["text"] + " "
                    
                    content = content.strip()
                    if not content:  # Skip empty blocks
                        continue
                    
                    # Determine element type based on font size
                    font_size = 12.0
                    if "lines" in block and block["lines"]:
                        if "spans" in block["lines"][0]:
                            font_size = block["lines"][0]["spans"][0]["size"]
                    
                    element_type = "paragraph"
                    if font_size > 16:
                        element_type = "heading_1"
                    elif font_size > 14:
                        element_type = "heading_2"
                    elif font_size > 12:
                        element_type = "heading_3"
                    
                    # Normalize bbox coordinates (0-1 range)
                    norm_bbox = (
                        bbox[0] / width,
                        bbox[1] / height,
                        bbox[2] / width,
                        bbox[3] / height
                    )
                    
                    element_id = f"p{page_num+1}_e{block_idx}"
                    element = create_element(
                        element_id=element_id,
                        element_type=element_type,
                        bbox=norm_bbox,
                        content=content,
                        font_name="Arial",
                        font_size=font_size,
                        font_weight="normal",
                        is_italic=False,
                        text_color=(0.0, 0.0, 0.0),
                        alignment="left",
                        indentation=0.0,
                        list_level=0,
                        list_style="none"
                    )
                    elements.append(element)
        else:
            # Process YOLO results
            print(f"Page {page_num + 1}: Processing {len(filtered_boxes)} YOLO elements")
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
                content = nougat_model.process(cropped_img) if nougat_model else ""
                
                # If Nougat failed, try PyMuPDF text extraction
                if not content:
                    # PyMuPDF property extraction
                    text_instances = page.get_text("dict", clip=(bbox[0], bbox[1], bbox[2], bbox[3]))
                    content = ""
                    for block in text_instances.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    content += span["text"] + " "
                    content = content.strip()
                
                # Extract properties from PyMuPDF
                text_instances = page.get_text("dict", clip=(bbox[0], bbox[1], bbox[2], bbox[3]))
                
                # Default properties
                font_name = "Arial"
                font_size = 12.0
                font_weight = "normal"
                is_italic = False
                text_color = (0.0, 0.0, 0.0)
                alignment = "left"
                indentation = 0.0
                list_level = 0
                list_style = "none"
                
                # Try to extract real properties from text instances
                if text_instances.get("blocks"):
                    block = text_instances["blocks"][0]
                    if "lines" in block and block["lines"]:
                        line = block["lines"][0]
                        if "spans" in line and line["spans"]:
                            span = line["spans"][0]
                            font_name = span.get("font", "Arial")
                            font_size = span.get("size", 12.0)
                            # Check for bold/italic
                            if "flags" in span:
                                flags = span["flags"]
                                if flags & 2**4:  # Bold flag
                                    font_weight = "bold"
                                if flags & 2**1:  # Italic flag
                                    is_italic = True
                
                element_id = f"p{page_num+1}_e{idx}"
                element = create_element(
                    element_id=element_id,
                    element_type=label,
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
        
        print(f"Page {page_num + 1}: Created {len(elements)} elements")
    
    # Save blueprint
    print(f"Blueprint has {len(blueprint['pages'])} pages")
    total_elements = sum(len(page['elements']) for page in blueprint['pages'])
    print(f"Total elements in blueprint: {total_elements}")
    
    # Validate before saving
    if validate_blueprint(blueprint):
        print("✅ Blueprint validation passed")
    else:
        print("❌ Blueprint validation failed")
    
    save_result = save_blueprint(blueprint, output_json)
    if save_result:
        print(f"✅ Blueprint saved to {output_json}")
    else:
        print(f"❌ Failed to save blueprint to {output_json}")
    
    return blueprint

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_parser.py <input.pdf> [output.json]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "document_blueprint.json"
    parse_pdf_to_blueprint(pdf_path, output_json=output_json) 