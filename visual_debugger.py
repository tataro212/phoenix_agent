"""
Visual Debugger: Verification Tool for Phoenix Agent
Draws colored, labeled boxes on page images to verify blueprint parsing results.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from document_blueprint import DocumentBlueprint, DocumentElement, ElementType


class VisualDebugger:
    """Visual verification tool for document blueprint parsing results."""
    
    # Color mapping for different element types
    ELEMENT_COLORS = {
        'paragraph': (255, 0, 0),      # Red
        'heading_1': (0, 255, 0),      # Green
        'heading_2': (0, 200, 0),      # Dark Green
        'heading_3': (0, 150, 0),      # Darker Green
        'list_item': (255, 165, 0),    # Orange
        'table': (0, 0, 255),          # Blue
        'image': (128, 0, 128),        # Purple
        'caption': (255, 255, 0),      # Yellow
        'footer': (128, 128, 128),     # Gray
        'header': (64, 64, 64),        # Dark Gray
        'toc_item': (0, 255, 255)      # Cyan
    }
    
    def __init__(self, output_dir: str = "debug_output"):
        """Initialize the visual debugger."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Try to load a font, fall back to default if not available
        try:
            self.font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                self.font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                self.font = ImageFont.load_default()
    
    def debug_blueprint(self, blueprint: DocumentBlueprint, page_images_dir: str) -> bool:
        """
        Creates visual debug images for each page in the blueprint.
        
        Args:
            blueprint: The document blueprint to debug
            page_images_dir: Directory containing page images
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            page_images_path = Path(page_images_dir)
            if not page_images_path.exists():
                print(f"Page images directory not found: {page_images_dir}")
                return False
            
            print(f"Creating visual debug images for {len(blueprint['pages'])} pages...")
            
            for page in blueprint['pages']:
                page_num = page['page_number']
                
                # Find the corresponding page image
                page_image_path = self._find_page_image(page_images_path, page_num)
                if not page_image_path:
                    print(f"Warning: No image found for page {page_num}")
                    continue
                
                # Create debug image
                debug_image = self._create_debug_image(page_image_path, page)
                
                # Save debug image
                output_path = self.output_dir / f"debug_page_{page_num:03d}.png"
                debug_image.save(output_path)
                print(f"Created debug image: {output_path}")
            
            # Create summary report
            self._create_summary_report(blueprint)
            
            print(f"Visual debugging complete. Output saved to: {self.output_dir}")
            return True
            
        except Exception as e:
            print(f"Error in visual debugging: {e}")
            return False
    
    def _find_page_image(self, images_dir: Path, page_num: int) -> Optional[Path]:
        """Find the image file corresponding to a page number."""
        # Common naming patterns
        patterns = [
            f"page_{page_num:03d}.png",
            f"page_{page_num:03d}.jpg",
            f"page_{page_num}.png",
            f"page_{page_num}.jpg",
            f"page-{page_num:03d}.png",
            f"page-{page_num:03d}.jpg",
            f"page-{page_num}.png",
            f"page-{page_num}.jpg",
            f"{page_num:03d}.png",
            f"{page_num:03d}.jpg",
            f"{page_num}.png",
            f"{page_num}.jpg"
        ]
        
        for pattern in patterns:
            image_path = images_dir / pattern
            if image_path.exists():
                return image_path
        
        # If no exact match, try to find any image file
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        if image_files:
            # Sort by name and return the one closest to page number
            image_files.sort(key=lambda x: x.name)
            if page_num <= len(image_files):
                return image_files[page_num - 1]
        
        return None
    
    def _create_debug_image(self, image_path: Path, page: dict) -> Image.Image:
        """Create a debug image with bounding boxes and labels."""
        # Load the original image
        original_image = Image.open(image_path)
        debug_image = original_image.copy()
        draw = ImageDraw.Draw(debug_image)
        
        # Draw each element
        for element in page['elements']:
            self._draw_element(draw, element, original_image.size)
        
        # Add page information
        self._draw_page_info(draw, page, original_image.size)
        
        return debug_image
    
    def _draw_element(self, draw: ImageDraw.Draw, element: DocumentElement, image_size: Tuple[int, int]) -> None:
        """Draw a single element with its bounding box and label."""
        bbox = element['bbox']
        element_type = element['type']
        
        # Convert normalized coordinates to pixel coordinates
        img_width, img_height = image_size
        x0 = int(bbox[0] * img_width)
        y0 = int(bbox[1] * img_height)
        x1 = int(bbox[2] * img_width)
        y1 = int(bbox[3] * img_height)
        
        # Get color for this element type
        color = self.ELEMENT_COLORS.get(element_type, (128, 128, 128))
        
        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # Draw label background
        label_text = f"{element_type}: {element['id']}"
        bbox_text = draw.textbbox((0, 0), label_text, font=self.font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Position label above the bounding box
        label_x = x0
        label_y = max(0, y0 - text_height - 5)
        
        # Draw label background
        draw.rectangle([
            label_x, label_y,
            label_x + text_width + 4, label_y + text_height + 4
        ], fill=(255, 255, 255), outline=color)
        
        # Draw label text
        draw.text((label_x + 2, label_y + 2), label_text, fill=color, font=self.font)
        
        # Draw children recursively
        for child in element['children']:
            self._draw_element(draw, child, image_size)
    
    def _draw_page_info(self, draw: ImageDraw.Draw, page: dict, image_size: Tuple[int, int]) -> None:
        """Draw page information in the top-left corner."""
        img_width, img_height = image_size
        
        info_text = f"Page {page['page_number']} | Elements: {len(page['elements'])}"
        bbox_text = draw.textbbox((0, 0), info_text, font=self.font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Draw info background
        draw.rectangle([
            10, 10,
            10 + text_width + 10, 10 + text_height + 10
        ], fill=(0, 0, 0, 128), outline=(255, 255, 255))
        
        # Draw info text
        draw.text((15, 15), info_text, fill=(255, 255, 255), font=self.font)
    
    def _create_summary_report(self, blueprint: DocumentBlueprint) -> None:
        """Create a summary report of the blueprint analysis."""
        report_path = self.output_dir / "debug_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PHOENIX AGENT - VISUAL DEBUG SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Document ID: {blueprint['metadata']['document_id']}\n")
            f.write(f"Source File: {blueprint['metadata']['source_file']}\n")
            f.write(f"Title: {blueprint['metadata']['title']}\n")
            f.write(f"Language: {blueprint['metadata']['language']}\n")
            f.write(f"Total Pages: {len(blueprint['pages'])}\n\n")
            
            # Element type statistics
            element_counts = {}
            total_elements = 0
            
            for page in blueprint['pages']:
                for element in page['elements']:
                    element_type = element['type']
                    element_counts[element_type] = element_counts.get(element_type, 0) + 1
                    total_elements += 1
            
            f.write("ELEMENT TYPE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            for element_type, count in sorted(element_counts.items()):
                percentage = (count / total_elements) * 100 if total_elements > 0 else 0
                f.write(f"{element_type:15s}: {count:4d} ({percentage:5.1f}%)\n")
            
            f.write(f"\nTotal Elements: {total_elements}\n\n")
            
            # Page-by-page breakdown
            f.write("PAGE-BY-PAGE BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            for page in blueprint['pages']:
                f.write(f"Page {page['page_number']:3d}: {len(page['elements']):3d} elements\n")
        
        print(f"Summary report created: {report_path}")
    
    def create_element_type_legend(self) -> None:
        """Create a legend showing all element types and their colors."""
        legend_path = self.output_dir / "element_legend.png"
        
        # Create legend image
        legend_height = len(self.ELEMENT_COLORS) * 30 + 50
        legend_image = Image.new('RGB', (400, legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend_image)
        
        y_offset = 20
        draw.text((20, y_offset), "ELEMENT TYPE LEGEND", fill=(0, 0, 0), font=self.font)
        y_offset += 30
        
        for element_type, color in self.ELEMENT_COLORS.items():
            # Draw color box
            draw.rectangle([20, y_offset, 50, y_offset + 20], fill=color, outline=(0, 0, 0))
            
            # Draw label
            draw.text((60, y_offset), element_type, fill=(0, 0, 0), font=self.font)
            
            y_offset += 25
        
        legend_image.save(legend_path)
        print(f"Element legend created: {legend_path}")


def debug_blueprint_from_file(blueprint_file: str, page_images_dir: str, output_dir: str = "debug_output") -> bool:
    """Convenience function to debug a blueprint from a JSON file."""
    try:
        from document_blueprint import load_blueprint
        
        # Load blueprint
        blueprint = load_blueprint(blueprint_file)
        if not blueprint:
            print(f"Failed to load blueprint from: {blueprint_file}")
            return False
        
        # Create debugger and run
        debugger = VisualDebugger(output_dir)
        success = debugger.debug_blueprint(blueprint, page_images_dir)
        
        if success:
            debugger.create_element_type_legend()
        
        return success
        
    except Exception as e:
        print(f"Error in debug_blueprint_from_file: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python visual_debugger.py <blueprint.json> <page_images_dir>")
        sys.exit(1)
    
    blueprint_file = sys.argv[1]
    page_images_dir = sys.argv[2]
    
    success = debug_blueprint_from_file(blueprint_file, page_images_dir)
    if success:
        print("Visual debugging completed successfully!")
    else:
        print("Visual debugging failed!")
        sys.exit(1) 