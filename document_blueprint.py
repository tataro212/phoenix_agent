"""
Document Blueprint: The System's DNA
The single source of truth for the entire Phoenix Agent system.
"""

from typing import TypedDict, List, Tuple, Literal, Optional, Dict, Any
import json
import copy
from pathlib import Path


# Defining precise types for clarity and validation
Alignment = Literal['left', 'center', 'right', 'justify']
ListStyle = Literal['bullet', 'numbered', 'none']
ElementType = Literal[
    'paragraph', 'heading_1', 'heading_2', 'heading_3',
    'list_item', 'table', 'image', 'caption', 'footer', 'header', 'toc_item'
]


class ElementProperties(TypedDict):
    """Holds all stylistic and semantic attributes of an element."""
    font_name: str
    font_size: float
    font_weight: Literal['normal', 'bold']
    is_italic: bool
    text_color: Tuple[float, float, float]  # RGB values from 0-1
    alignment: Alignment
    indentation: float  # First line indentation
    list_level: int  # 0 for non-list, 1 for top-level, etc.
    list_style: ListStyle


class DocumentElement(TypedDict):
    """The core building block of the document."""
    id: str
    type: ElementType
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    content: str  # Raw text, markdown from Nougat, or image path
    properties: ElementProperties
    children: List['DocumentElement']  # For nested structures like table cells


class Page(TypedDict):
    """Represents a single page in the document."""
    page_number: int
    dimensions: Tuple[float, float]  # (width, height)
    elements: List[DocumentElement]
    reading_order: List[str]  # List of element 'id's in order


class DocumentMetadata(TypedDict):
    """Metadata for the entire document."""
    document_id: str
    source_file: str
    title: str
    language: str


class DocumentBlueprint(TypedDict):
    """The single source of truth for the entire document."""
    metadata: DocumentMetadata
    pages: List[Page]


def create_blueprint(file_path: str) -> DocumentBlueprint:
    """Initializes a new, empty blueprint."""
    source_path = Path(file_path)
    
    blueprint: DocumentBlueprint = {
        "metadata": {
            "document_id": f"doc_{source_path.stem}_{int(Path(file_path).stat().st_mtime)}",
            "source_file": str(source_path.absolute()),
            "title": source_path.stem,
            "language": "unknown"
        },
        "pages": []
    }
    
    return blueprint


def validate_blueprint(data: dict) -> bool:
    """Validates a dictionary against the DocumentBlueprint schema."""
    try:
        # Check top-level structure
        if not isinstance(data, dict):
            return False
        
        if "metadata" not in data or "pages" not in data:
            return False
        
        # Validate metadata
        metadata = data["metadata"]
        required_metadata_fields = ["document_id", "source_file", "title", "language"]
        for field in required_metadata_fields:
            if field not in metadata or not isinstance(metadata[field], str):
                return False
        
        # Validate pages
        if not isinstance(data["pages"], list):
            return False
        
        for page in data["pages"]:
            if not validate_page(page):
                return False
        
        return True
        
    except Exception:
        return False


def validate_page(page: dict) -> bool:
    """Validates a single page structure."""
    try:
        if not isinstance(page, dict):
            return False
        
        required_page_fields = ["page_number", "dimensions", "elements", "reading_order"]
        for field in required_page_fields:
            if field not in page:
                return False
        
        # Validate page_number
        if not isinstance(page["page_number"], int):
            return False
        
        # Validate dimensions
        dimensions = page["dimensions"]
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            return False
        if not all(isinstance(d, (int, float)) for d in dimensions):
            return False
        
        # Validate elements
        if not isinstance(page["elements"], list):
            return False
        
        for element in page["elements"]:
            if not validate_element(element):
                return False
        
        # Validate reading_order
        reading_order = page["reading_order"]
        if not isinstance(reading_order, list):
            return False
        if not all(isinstance(item, str) for item in reading_order):
            return False
        
        return True
        
    except Exception:
        return False


def validate_element(element: dict) -> bool:
    """Validates a single element structure."""
    try:
        if not isinstance(element, dict):
            return False
        
        required_element_fields = ["id", "type", "bbox", "content", "properties", "children"]
        for field in required_element_fields:
            if field not in element:
                return False
        
        # Validate id
        if not isinstance(element["id"], str):
            return False
        
        # Validate type
        valid_types = [
            'paragraph', 'heading_1', 'heading_2', 'heading_3',
            'list_item', 'table', 'image', 'caption', 'footer', 'header', 'toc_item'
        ]
        if element["type"] not in valid_types:
            return False
        
        # Validate bbox
        bbox = element["bbox"]
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            return False
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            return False
        
        # Validate content
        if not isinstance(element["content"], str):
            return False
        
        # Validate properties
        if not validate_properties(element["properties"]):
            return False
        
        # Validate children
        if not isinstance(element["children"], list):
            return False
        for child in element["children"]:
            if not validate_element(child):
                return False
        
        return True
        
    except Exception:
        return False


def validate_properties(properties: dict) -> bool:
    """Validates element properties structure."""
    try:
        if not isinstance(properties, dict):
            return False
        
        required_property_fields = [
            "font_name", "font_size", "font_weight", "is_italic", 
            "text_color", "alignment", "indentation", "list_level", "list_style"
        ]
        for field in required_property_fields:
            if field not in properties:
                return False
        
        # Validate font_name
        if not isinstance(properties["font_name"], str):
            return False
        
        # Validate font_size
        if not isinstance(properties["font_size"], (int, float)):
            return False
        
        # Validate font_weight
        if properties["font_weight"] not in ["normal", "bold"]:
            return False
        
        # Validate is_italic
        if not isinstance(properties["is_italic"], bool):
            return False
        
        # Validate text_color
        text_color = properties["text_color"]
        if not isinstance(text_color, tuple) or len(text_color) != 3:
            return False
        if not all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in text_color):
            return False
        
        # Validate alignment
        if properties["alignment"] not in ["left", "center", "right", "justify"]:
            return False
        
        # Validate indentation
        if not isinstance(properties["indentation"], (int, float)):
            return False
        
        # Validate list_level
        if not isinstance(properties["list_level"], int):
            return False
        
        # Validate list_style
        if properties["list_style"] not in ["bullet", "numbered", "none"]:
            return False
        
        return True
        
    except Exception:
        return False


def save_blueprint(blueprint: DocumentBlueprint, file_path: str) -> bool:
    """Saves a blueprint to JSON file with validation."""
    try:
        if not validate_blueprint(blueprint):
            raise ValueError("Invalid blueprint structure")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(blueprint, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error saving blueprint: {e}")
        return False


def load_blueprint(file_path: str) -> Optional[DocumentBlueprint]:
    """Loads a blueprint from JSON file with validation."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not validate_blueprint(data):
            raise ValueError("Invalid blueprint structure in file")
        
        return data
        
    except Exception as e:
        print(f"Error loading blueprint: {e}")
        return None


def find_element_by_id(blueprint: DocumentBlueprint, element_id: str) -> Optional[DocumentElement]:
    """Finds an element by its ID across all pages."""
    for page in blueprint["pages"]:
        for element in page["elements"]:
            if element["id"] == element_id:
                return element
            # Check children recursively
            found = find_element_in_children(element, element_id)
            if found:
                return found
    return None


def find_element_in_children(element: DocumentElement, element_id: str) -> Optional[DocumentElement]:
    """Recursively searches for an element in children."""
    for child in element["children"]:
        if child["id"] == element_id:
            return child
        found = find_element_in_children(child, element_id)
        if found:
            return found
    return None


def create_element(
    element_id: str,
    element_type: ElementType,
    bbox: Tuple[float, float, float, float],
    content: str,
    font_name: str = "Arial",
    font_size: float = 12.0,
    font_weight: Literal['normal', 'bold'] = 'normal',
    is_italic: bool = False,
    text_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    alignment: Alignment = 'left',
    indentation: float = 0.0,
    list_level: int = 0,
    list_style: ListStyle = 'none'
) -> DocumentElement:
    """Factory function to create a properly structured DocumentElement."""
    properties: ElementProperties = {
        "font_name": font_name,
        "font_size": font_size,
        "font_weight": font_weight,
        "is_italic": is_italic,
        "text_color": text_color,
        "alignment": alignment,
        "indentation": indentation,
        "list_level": list_level,
        "list_style": list_style
    }
    
    element: DocumentElement = {
        "id": element_id,
        "type": element_type,
        "bbox": bbox,
        "content": content,
        "properties": properties,
        "children": []
    }
    
    return element


def create_page(page_number: int, width: float, height: float) -> Page:
    """Factory function to create a properly structured Page."""
    page: Page = {
        "page_number": page_number,
        "dimensions": (width, height),
        "elements": [],
        "reading_order": []
    }
    
    return page 