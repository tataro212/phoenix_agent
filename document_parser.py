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
    DocumentBlueprint, create_blueprint, create_element, create_page, save_blueprint, validate_blueprint, validate_properties
)

# Try to import torch for Nougat
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Install with: pip install torch")

# Try to import ultralytics for YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Try to import Nougat
try:
    from transformers import AutoTokenizer, AutoModelForVision2Seq
    from nougat import NougatModel
    NOUGAT_AVAILABLE = True
except ImportError:
    NOUGAT_AVAILABLE = False
    print("Warning: nougat-ocr not available. Install with: pip install nougat-ocr")

# --- YOLO Model Integration ---
class YOLOModel:
    def __init__(self, model_path: str = None, allow_fallback: bool = False):
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
            print(f"❌ Error loading YOLO model {model_path}: {e}")
            if allow_fallback:
                print("⚠️  Using YOLOv8n as fallback.")
                self.model = YOLO("yolov8n.pt")
            else:
                raise RuntimeError(f"Failed to load YOLO model and fallback is not allowed. Exiting.")
    
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
            print("⚠️  Nougat not available. Using PyMuPDF text extraction as fallback.")
            self.model = None
            self.tokenizer = None
            return
        
        try:
            if model_path is None:
                model_path = "facebook/nougat-base"
            
            # Load tokenizer first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"✅ Loaded Nougat tokenizer: {model_path}")
            except Exception as e:
                print(f"⚠️  Nougat tokenizer loading failed: {e}")
                print("⚠️  Will use PyMuPDF text extraction as fallback")
                self.model = None
                self.tokenizer = None
                return
            
            # Load model
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(model_path)
                print(f"✅ Loaded Nougat model: {model_path}")
            except Exception as e:
                print(f"⚠️  Nougat model loading failed: {e}")
                print("⚠️  Will use PyMuPDF text extraction as fallback")
                self.model = None
                self.tokenizer = None
                return
                
        except Exception as e:
            print(f"⚠️  Nougat model loading failed: {e}")
            print("⚠️  Will use PyMuPDF text extraction as fallback")
            self.model = None
            self.tokenizer = None
    
    def process(self, image: np.ndarray) -> str:
        """Extract text from image using Nougat."""
        if self.model is None or self.tokenizer is None or not TORCH_AVAILABLE:
            # Fallback to basic OCR or return empty string
            return ""
        
        try:
            # Convert image to PIL Image for Nougat
            from PIL import Image
            pil_image = Image.fromarray(image)
            
            # Process with Nougat using the proper API
            # Note: This is a simplified version - in production you'd want more robust processing
            inputs = self.tokenizer(pil_image, return_tensors="pt")
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=512)
            
            # Decode the output
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result if result else ""
            
        except Exception as e:
            print(f"⚠️  Error in Nougat processing: {e}")
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
    
    # Load models with strict error handling
    try:
        yolo_model = YOLOModel(yolo_model_path, allow_fallback=False)
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        print("❌ Cannot proceed without YOLO model. Exiting.")
        sys.exit(1)
    
    try:
        nougat_model = NougatModel(nougat_model_path)
        if nougat_model.model is not None:
            print("✅ Nougat model loaded successfully")
        else:
            print("⚠️  Nougat model loading failed, will use PyMuPDF fallback")
    except Exception as e:
        print(f"⚠️  Nougat model loading failed: {e}")
        print("⚠️  Will use PyMuPDF text extraction as fallback")
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
        
        # Apply NMS and track dropped elements
        filtered_boxes = []
        total_dropped = 0
        nms_threshold = 0.5
        for label, boxes in class_to_boxes.items():
            scores = class_to_scores[label]
            keep = non_max_suppression(boxes, scores, iou_threshold=nms_threshold)
            dropped_count = len(boxes) - len(keep)
            total_dropped += dropped_count
            
            if dropped_count > 0:
                print(f"Page {page_num + 1}: Dropped {dropped_count} {label} elements due to NMS (IoU threshold: {nms_threshold})")
            
            for idx in keep:
                filtered_boxes.append({"bbox": boxes[idx], "class_label": label, "confidence": scores[idx]})
        
        if total_dropped > 0:
            print(f"Page {page_num + 1}: Total dropped elements after NMS: {total_dropped}")
        
        # Confidence threshold filtering (example: 0.5)
        conf_threshold = 0.5
        before_conf_filter = len(filtered_boxes)
        filtered_boxes = [b for b in filtered_boxes if b["confidence"] >= conf_threshold]
        dropped_conf = before_conf_filter - len(filtered_boxes)
        if dropped_conf > 0:
            print(f"Page {page_num + 1}: Dropped {dropped_conf} elements due to low confidence (< {conf_threshold})")
        
        print(f"Page {page_num + 1}: Processing {len(filtered_boxes)} remaining elements after NMS and confidence filtering")
        
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
                    
                    # Extract properties from the first span
                    font_name = "Arial"
                    font_size = 12.0
                    font_weight = "normal"
                    is_italic = False
                    text_color = (0.0, 0.0, 0.0)
                    alignment = "left"
                    indentation = 0.0
                    list_level = 0
                    list_style = "none"
                    
                    if "lines" in block and block["lines"]:
                        if "spans" in block["lines"][0]:
                            span = block["lines"][0]["spans"][0]
                            font_name = span.get("font", "Arial")
                            font_size = span.get("size", 12.0)
                            
                            # Extract color
                            color_int = span.get("color", 0)
                            text_color = (
                                ((color_int >> 16) & 255) / 255.0,
                                ((color_int >> 8) & 255) / 255.0,
                                (color_int & 255) / 255.0
                            )
                            
                            # Check for bold/italic from flags
                            if "flags" in span:
                                flags = span["flags"]
                                if flags & 2**4:  # Bold flag
                                    font_weight = "bold"
                                if flags & 2**1:  # Italic flag
                                    is_italic = True
                    
                    # Determine element type based on font size
                    element_type = "paragraph"
                    if font_size > 16:
                        element_type = "heading_1"
                    elif font_size > 14:
                        element_type = "heading_2"
                    elif font_size > 12:
                        element_type = "heading_3"
                    
                    # Normalize bbox coordinates (0-1 range)
                    norm_bbox = (
                        float(bbox[0]) / float(width),
                        float(bbox[1]) / float(height),
                        float(bbox[2]) / float(width),
                        float(bbox[3]) / float(height)
                    )
                    
                    # Enhanced alignment detection
                    alignment = detect_text_alignment(block, bbox, width)
                    
                    # Enhanced indentation detection
                    indentation = detect_text_indentation(block, bbox, width)
                    
                    # Enhanced list detection
                    list_level, list_style = detect_list_structure(block, content, font_size)
                    
                    # After calculating indentation, ensure it's a float and valid
                    try:
                        indentation = float(indentation)
                        if indentation != indentation or indentation == float('inf') or indentation == float('-inf'):
                            indentation = 0.0
                    except Exception:
                        indentation = 0.0
                    
                    element_id = f"p{page_num+1}_e{block_idx}"
                    element = create_element(
                        element_id=element_id,
                        element_type=element_type,
                        bbox=norm_bbox,
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
                
                # Nougat OCR attempt
                content = ""
                if nougat_model:
                    try:
                        content = nougat_model.process(cropped_img)
                        if content:
                            print(f"-> Nougat OCR successful for element p{page_num+1}_e{idx}")
                    except Exception as e:
                        print(f"⚠️ Nougat processing failed for element {idx}: {e}")
                        content = ""  # Ensure content is empty to trigger fallback

                # --- START: ADD THIS ROBUST FALLBACK ---
                # If Nougat was not used or failed, use PyMuPDF for this specific box
                if not content:
                    print(f"-> Falling back to PyMuPDF for element p{page_num+1}_e{idx}")
                    # The bbox is in image (pixel) coordinates, convert to PDF points
                    # Note: zoom = 300/72 = 4.1667, so we divide by zoom to get PDF coordinates
                    pdf_bbox = (
                        bbox[0] / zoom, 
                        bbox[1] / zoom, 
                        bbox[2] / zoom, 
                        bbox[3] / zoom
                    )
                    # Extract text ONLY from within the detected bounding box
                    try:
                        content = page.get_text("text", clip=pdf_bbox, sort=True).strip()
                        if not content:
                            # Try alternative extraction method
                            content = page.get_text("dict", clip=pdf_bbox)
                            if content and "blocks" in content:
                                text_parts = []
                                for block in content["blocks"]:
                                    if "lines" in block:
                                        for line in block["lines"]:
                                            for span in line["spans"]:
                                                text_parts.append(span["text"])
                                content = " ".join(text_parts).strip()
                        print(f"-> PyMuPDF fallback extracted {len(content)} characters for element p{page_num+1}_e{idx}")
                    except Exception as e:
                        print(f"-> PyMuPDF fallback failed for element p{page_num+1}_e{idx}: {e}")
                        content = ""
                # --- END: ROBUST FALLBACK ---
                
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
                            
                            # Extract color
                            color_int = span.get("color", 0)
                            text_color = (
                                ((color_int >> 16) & 255) / 255.0,
                                ((color_int >> 8) & 255) / 255.0,
                                (color_int & 255) / 255.0
                            )
                            
                            # Check for bold/italic from flags
                            if "flags" in span:
                                flags = span["flags"]
                                if flags & 2**4:  # Bold flag
                                    font_weight = "bold"
                                if flags & 2**1:  # Italic flag
                                    is_italic = True
                
                # Enhanced property detection (only if we have a valid block)
                if text_instances.get("blocks") and text_instances["blocks"]:
                    block = text_instances["blocks"][0]
                    # Enhanced alignment detection
                    alignment = detect_text_alignment(block, bbox, width)
                    
                    # Enhanced indentation detection
                    indentation = detect_text_indentation(block, bbox, width)
                    
                    # Enhanced list detection
                    list_level, list_style = detect_list_structure(block, content, font_size)
                else:
                    # Use defaults if no block data available
                    alignment = "left"
                    indentation = 0.0
                    list_level = 0
                    list_style = "none"
                
                # After calculating indentation, ensure it's a float and valid
                try:
                    indentation = float(indentation)
                    if indentation != indentation or indentation == float('inf') or indentation == float('-inf'):
                        indentation = 0.0
                except Exception:
                    indentation = 0.0
                
                element_id = f"p{page_num+1}_e{idx}"
                element = create_element(
                    element_id=element_id,
                    element_type=label,
                    bbox=(float(bbox[0])/float(img.shape[1]), float(bbox[1])/float(img.shape[0]), float(bbox[2])/float(img.shape[1]), float(bbox[3])/float(img.shape[0])),
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
        # Improved reading order detection for multi-column layouts
        if elements:
            # Sort elements by position using a more sophisticated algorithm
            sorted_elements = determine_reading_order(elements, width, height)
            page_obj["reading_order"] = [el["id"] for el in sorted_elements]
        else:
            page_obj["reading_order"] = []
        
        blueprint["pages"].append(page_obj)
        
        print(f"Page {page_num + 1}: Created {len(elements)} elements")
    
    # Save blueprint
    print(f"Blueprint has {len(blueprint['pages'])} pages")
    total_elements = sum(len(page['elements']) for page in blueprint['pages'])
    print(f"Total elements in blueprint: {total_elements}")
    
    # Validate before saving with detailed debugging
    print("\n🔍 DEBUGGING BLUEPRINT VALIDATION:")
    print("="*50)
    
    # Check top-level structure
    if not isinstance(blueprint, dict):
        print("❌ Blueprint is not a dict")
        return blueprint
    
    if "metadata" not in blueprint or "pages" not in blueprint:
        print("❌ Missing metadata or pages in blueprint")
        return blueprint
    
    # Validate metadata
    metadata = blueprint["metadata"]
    required_metadata_fields = ["document_id", "source_file", "title", "language"]
    for field in required_metadata_fields:
        if field not in metadata or not isinstance(metadata[field], str):
            print(f"❌ Invalid metadata field: {field}")
            return blueprint
    
    # Validate pages
    if not isinstance(blueprint["pages"], list):
        print("❌ Pages is not a list")
        return blueprint
    
    # Check each page
    for page_idx, page in enumerate(blueprint["pages"]):
        print(f"\n📄 Validating page {page_idx + 1}:")
        
        if not isinstance(page, dict):
            print(f"❌ Page {page_idx + 1} is not a dict")
            return blueprint
        
        required_page_fields = ["page_number", "dimensions", "elements", "reading_order"]
        for field in required_page_fields:
            if field not in page:
                print(f"❌ Page {page_idx + 1} missing field: {field}")
                return blueprint
        
        # Validate page_number
        if not isinstance(page["page_number"], int):
            print(f"❌ Page {page_idx + 1} page_number is not int")
            return blueprint
        
        # Validate dimensions
        dimensions = page["dimensions"]
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            print(f"❌ Page {page_idx + 1} dimensions invalid")
            return blueprint
        if not all(isinstance(d, (int, float)) for d in dimensions):
            print(f"❌ Page {page_idx + 1} dimensions not numeric")
            return blueprint
        
        # Validate elements
        if not isinstance(page["elements"], list):
            print(f"❌ Page {page_idx + 1} elements is not a list")
            return blueprint
        
        # Check each element
        for elem_idx, element in enumerate(page["elements"]):
            print(f"  🔍 Element {elem_idx + 1}: {element.get('id', 'NO_ID')} - {element.get('type', 'NO_TYPE')}")
            
            if not isinstance(element, dict):
                print(f"❌ Element {elem_idx + 1} is not a dict")
                return blueprint
            
            required_element_fields = ["id", "type", "bbox", "content", "properties", "children"]
            for field in required_element_fields:
                if field not in element:
                    print(f"❌ Element {elem_idx + 1} missing field: {field}")
                    return blueprint
            
            # Validate id
            if not isinstance(element["id"], str):
                print(f"❌ Element {elem_idx + 1} id is not string")
                return blueprint
            
            # Validate type
            valid_types = [
                'paragraph', 'heading_1', 'heading_2', 'heading_3',
                'list_item', 'table', 'image', 'caption', 'footer', 'header', 'toc_item'
            ]
            if element["type"] not in valid_types:
                print(f"❌ Element {elem_idx + 1} invalid type: {element['type']}")
                return blueprint
            
            # Validate bbox
            bbox = element["bbox"]
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                print(f"❌ Element {elem_idx + 1} invalid bbox")
                return blueprint
            if not all(isinstance(coord, (int, float)) for coord in bbox):
                print(f"❌ Element {elem_idx + 1} bbox not numeric")
                return blueprint
            
            # Validate content
            if not isinstance(element["content"], str):
                print(f"❌ Element {elem_idx + 1} content is not string")
                return blueprint
            
            # Validate properties
            if not validate_properties(element["properties"], element["id"]):
                print(f"❌ Element {elem_idx + 1} properties validation failed")
                return blueprint
            
            # Validate children
            if not isinstance(element["children"], list):
                print(f"❌ Element {elem_idx + 1} children is not list")
                return blueprint
        
        # Validate reading_order
        reading_order = page["reading_order"]
        if not isinstance(reading_order, list):
            print(f"❌ Page {page_idx + 1} reading_order is not list")
            return blueprint
        if not all(isinstance(item, str) for item in reading_order):
            print(f"❌ Page {page_idx + 1} reading_order not all strings")
            return blueprint
    
    print("✅ All validation checks passed!")
    print("="*50)
    
    # Now try the actual validation
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

def determine_reading_order(elements: List[Dict], page_width: float, page_height: float) -> List[Dict]:
    """
    Determines reading order for elements using an advanced multi-column aware algorithm.
    Handles complex layouts including mixed content types and irregular column structures.
    """
    if not elements:
        return []
    
    # Normalize coordinates if they're not already normalized
    normalized_elements = []
    for element in elements:
        bbox = element["bbox"]
        if bbox[0] > 1.0 or bbox[1] > 1.0:  # Not normalized
            norm_bbox = (
                bbox[0] / page_width,
                bbox[1] / page_height,
                bbox[2] / page_width,
                bbox[3] / page_height
            )
            element_copy = element.copy()
            element_copy["bbox"] = norm_bbox
            normalized_elements.append(element_copy)
        else:
            normalized_elements.append(element)
    
    # Enhanced column detection using clustering
    x_centers = [(el["bbox"][0] + el["bbox"][2]) / 2 for el in normalized_elements]
    
    # Use K-means clustering to detect columns more robustly
    columns = detect_columns_using_clustering(normalized_elements, x_centers)
    
    # If clustering fails or detects only one column, use fallback method
    if len(columns) <= 1:
        return detect_reading_order_fallback(normalized_elements)
    
    # Sort each column by y-coordinate with tolerance for slight overlaps
    for column in columns:
        column.sort(key=lambda el: el["bbox"][1])
    
    # Merge columns using advanced reading order algorithm
    result = merge_columns_advanced(columns, page_height)
    
    return result

def detect_columns_using_clustering(elements: List[Dict], x_centers: List[float]) -> List[List[Dict]]:
    """
    Uses K-means clustering to detect columns more robustly.
    """
    if len(elements) < 2:
        return [elements]
    
    # Try different numbers of clusters (1 to min(5, num_elements))
    best_clusters = None
    best_score = float('inf')
    
    for k in range(1, min(6, len(elements) + 1)):
        try:
            clusters = kmeans_clustering(x_centers, k)
            score = calculate_clustering_score(clusters, x_centers)
            if score < best_score:
                best_score = score
                best_clusters = clusters
        except:
            continue
    
    if best_clusters is None:
        return [elements]
    
    # Group elements by cluster
    columns = [[] for _ in range(len(best_clusters))]
    for i, cluster_id in enumerate(best_clusters):
        columns[cluster_id].append(elements[i])
    
    # Filter out empty columns
    columns = [col for col in columns if col]
    
    return columns

def kmeans_clustering(data: List[float], k: int) -> List[int]:
    """
    Simple K-means clustering implementation.
    """
    if len(data) == 0:
        return []
    
    # Initialize centroids
    centroids = [data[i] for i in range(0, len(data), max(1, len(data) // k))][:k]
    
    for _ in range(10):  # Max 10 iterations
        # Assign points to nearest centroid
        clusters = []
        for point in data:
            distances = [abs(point - c) for c in centroids]
            clusters.append(distances.index(min(distances)))
        
        # Update centroids
        new_centroids = []
        for i in range(k):
            cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]
            if cluster_points:
                new_centroids.append(sum(cluster_points) / len(cluster_points))
            else:
                new_centroids.append(centroids[i])
        
        # Check convergence
        if all(abs(new_centroids[i] - centroids[i]) < 0.01 for i in range(k)):
            break
        
        centroids = new_centroids
    
    return clusters

def calculate_clustering_score(clusters: List[int], data: List[float]) -> float:
    """
    Calculate the quality of clustering (lower is better).
    """
    if not clusters:
        return float('inf')
    
    # Calculate within-cluster variance
    total_variance = 0
    for cluster_id in set(clusters):
        cluster_points = [data[i] for i in range(len(data)) if clusters[i] == cluster_id]
        if len(cluster_points) > 1:
            mean_val = sum(cluster_points) / len(cluster_points)
            variance = sum((p - mean_val) ** 2 for p in cluster_points) / len(cluster_points)
            total_variance += variance
    
    return total_variance

def detect_reading_order_fallback(elements: List[Dict]) -> List[Dict]:
    """
    Fallback reading order detection for single-column or simple layouts.
    """
    # Sort by y-coordinate first, then by x-coordinate for elements at similar heights
    return sorted(elements, key=lambda el: (el["bbox"][1], el["bbox"][0]))

def merge_columns_advanced(columns: List[List[Dict]], page_height: float) -> List[Dict]:
    """
    Advanced column merging algorithm that handles complex layouts.
    """
    if not columns:
        return []
    
    # Calculate the height range of each column
    column_ranges = []
    for column in columns:
        if column:
            min_y = min(el["bbox"][1] for el in column)
            max_y = max(el["bbox"][3] for el in column)
            column_ranges.append((min_y, max_y))
        else:
            column_ranges.append((0, 0))
    
    # Determine if columns have similar height ranges (likely same content flow)
    similar_heights = all(
        abs(ranges[1] - ranges[0]) < 0.1  # 10% tolerance
        for ranges in column_ranges
    )
    
    if similar_heights:
        # Use row-by-row merging for similar height columns
        return merge_columns_row_by_row(columns)
    else:
        # Use sequential merging for different height columns
        return merge_columns_sequential(columns)

def merge_columns_row_by_row(columns: List[List[Dict]]) -> List[Dict]:
    """
    Merge columns row by row (left to right, top to bottom).
    """
    result = []
    max_column_length = max(len(col) for col in columns)
    
    for row_idx in range(max_column_length):
        for column in columns:
            if row_idx < len(column):
                result.append(column[row_idx])
    
    return result

def merge_columns_sequential(columns: List[List[Dict]]) -> List[Dict]:
    """
    Merge columns sequentially (left column first, then next column).
    """
    result = []
    for column in columns:
        result.extend(column)
    
    return result

def detect_text_alignment(block: Dict, bbox: Tuple[float, float, float, float], page_width: float) -> str:
    """
    Detect text alignment based on block positioning and line analysis.
    """
    if "lines" not in block or not block["lines"]:
        return "left"
    
    # Analyze all lines in the block
    line_positions = []
    for line in block["lines"]:
        if "spans" in line and line["spans"]:
            # Calculate line bounds
            line_x0 = min(span.get("bbox", [0, 0, 0, 0])[0] for span in line["spans"])
            line_x1 = max(span.get("bbox", [0, 0, 0, 0])[2] for span in line["spans"])
            line_width = line_x1 - line_x0
            line_center = (line_x0 + line_x1) / 2
            line_positions.append((line_center, line_width))
    
    if not line_positions:
        return "left"
    
    # Calculate block center and width
    block_center = (bbox[0] + bbox[2]) / 2
    block_width = bbox[2] - bbox[0]
    
    # Analyze alignment patterns
    left_aligned = 0
    center_aligned = 0
    right_aligned = 0
    
    for line_center, line_width in line_positions:
        # Normalize positions
        norm_line_center = line_center / page_width
        norm_block_center = block_center / page_width
        norm_block_width = block_width / page_width
        
        # Check alignment with tolerance
        tolerance = 0.05  # 5% of page width
        
        if abs(norm_line_center - norm_block_center) < tolerance:
            center_aligned += 1
        elif norm_line_center < norm_block_center - tolerance:
            left_aligned += 1
        else:
            right_aligned += 1
    
    # Determine dominant alignment
    total_lines = len(line_positions)
    if center_aligned / total_lines > 0.6:
        return "center"
    elif right_aligned / total_lines > 0.6:
        return "right"
    else:
        return "left"

def detect_text_indentation(block: Dict, bbox: Tuple[float, float, float, float], page_width: float) -> float:
    """
    Detect text indentation based on first line positioning.
    """
    if "lines" not in block or not block["lines"]:
        return 0.0
    # Get the first line
    first_line = block["lines"][0]
    if "spans" not in first_line or not first_line["spans"]:
        return 0.0
    # Calculate first line start position
    first_span = first_line["spans"][0]
    first_line_start = first_span.get("bbox", [0, 0, 0, 0])[0]
    # Calculate indentation relative to block start
    indentation = first_line_start - bbox[0]
    # Normalize to page width
    try:
        norm_indentation = float(indentation) / float(page_width) if page_width else 0.0
    except Exception:
        norm_indentation = 0.0
    # Return indentation in points (assuming 72 DPI)
    result = max(0.0, norm_indentation * 72.0)
    if not isinstance(result, float) or result != result or result == float('inf') or result == float('-inf'):
        return 0.0
    return result

def detect_list_structure(block: Dict, content: str, font_size: float) -> Tuple[int, str]:
    """
    Detect list structure based on content analysis and formatting.
    """
    # Check for common list indicators
    list_indicators = [
        r'^\s*[\u2022\u25E6\u25AA\u25AB\u25CB\u25CF\u25D8\u25D9]\s+',  # Bullet points
        r'^\s*\d+\.\s+',  # Numbered lists
        r'^\s*[a-zA-Z]\.\s+',  # Lettered lists
        r'^\s*[ivxlcdm]+\.\s+',  # Roman numerals
    ]
    
    import re
    
    for pattern in list_indicators:
        if re.match(pattern, content, re.IGNORECASE):
            # Determine list level based on indentation and font size
            list_level = 1
            
            # Check for nested lists (multiple levels of indentation)
            if re.match(r'^\s{4,}', content):  # 4+ spaces or equivalent
                list_level = 2
            elif re.match(r'^\s{8,}', content):  # 8+ spaces or equivalent
                list_level = 3
            
            # Determine list style
            if re.match(r'^\s*\d+\.\s+', content):
                list_style = "numbered"
            else:
                list_style = "bullet"
            
            return list_level, list_style
    
    # Check for implicit list structure (repeated patterns)
    lines = content.split('\n')
    if len(lines) > 1:
        # Check if multiple lines have similar structure
        similar_structure = True
        first_line_start = None
        
        for line in lines:
            if line.strip():
                # Find the first non-whitespace character
                match = re.match(r'^(\s*)', line)
                if match:
                    indent = len(match.group(1))
                    if first_line_start is None:
                        first_line_start = indent
                    elif abs(indent - first_line_start) > 2:  # Allow small variations
                        similar_structure = False
                        break
        
        if similar_structure and len(lines) >= 2:
            return 1, "bullet"
    
    return 0, "none"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_parser.py <input.pdf> [output.json]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "document_blueprint.json"
    parse_pdf_to_blueprint(pdf_path, output_json=output_json) 