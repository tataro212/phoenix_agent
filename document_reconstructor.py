"""
Station 3: The Architect (document_reconstructor.py)
Transforms the abstract translated_blueprint.json into a concrete, high-fidelity PDF file.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.colors import black, white
from reportlab.pdfgen import canvas
from document_blueprint import load_blueprint, DocumentBlueprint, DocumentElement

# Define point manually if not available
point = inch / 72.0

class DocumentReconstructor:
    """The Architect: Blueprint-driven PDF reconstruction engine."""
    
    def __init__(self, blueprint: DocumentBlueprint):
        self.blueprint = blueprint
        self.element_page_map = {}  # Maps element IDs to page numbers
        self.current_page = 1
        
        # Base indentation for list levels
        self.BASE_INDENT = 20
        self.BULLET_CHARS = ['\u2022', '\u25E6', '\u25AA']  # Bullet, circle, square
        
        # Page dimensions from blueprint
        if self.blueprint['pages']:
            first_page = self.blueprint['pages'][0]
            self.page_width = float(first_page['dimensions'][0])
            self.page_height = float(first_page['dimensions'][1])
        else:
            self.page_width, self.page_height = letter
    
    def create_paragraph_style(self, element: DocumentElement) -> ParagraphStyle:
        """Creates a ReportLab ParagraphStyle from blueprint element properties."""
        properties = element['properties']
        
        # Map alignment from blueprint to ReportLab constants
        alignment_map = {
            'left': TA_LEFT,
            'center': TA_CENTER,
            'right': TA_RIGHT,
            'justify': TA_JUSTIFY
        }
        
        # Map font names to standard ReportLab fonts
        font_map = {
            'arial': 'Helvetica',
            'Arial': 'Helvetica',
            'times': 'Times-Roman',
            'Times': 'Times-Roman',
            'times new roman': 'Times-Roman',
            'Times New Roman': 'Times-Roman',
            'courier': 'Courier',
            'Courier': 'Courier',
            'courier new': 'Courier',
            'Courier New': 'Courier'
        }
        
        # Get the mapped font name or use Helvetica as default
        font_name = font_map.get(properties['font_name'].lower(), 'Helvetica')
        
        style = ParagraphStyle(
            name=f"style_{element['id']}",
            fontName=font_name,
            fontSize=properties['font_size'],
            alignment=alignment_map.get(properties['alignment'], TA_LEFT),
            firstLineIndent=properties['indentation'],
            textColor=properties['text_color'],
            spaceAfter=6,
            spaceBefore=6
        )
        
        # Handle bold and italic - use standard font variants
        if properties['font_weight'] == 'bold':
            if font_name == 'Helvetica':
                style.fontName = 'Helvetica-Bold'
            elif font_name == 'Times-Roman':
                style.fontName = 'Times-Bold'
            elif font_name == 'Courier':
                style.fontName = 'Courier-Bold'
        if properties['is_italic']:
            if font_name == 'Helvetica':
                style.fontName = 'Helvetica-Oblique'
            elif font_name == 'Times-Roman':
                style.fontName = 'Times-Italic'
            elif font_name == 'Courier':
                style.fontName = 'Courier-Oblique'
        
        # Handle list formatting
        if element['type'] == 'list_item':
            list_level = properties['list_level']
            style.leftIndent = list_level * self.BASE_INDENT
            style.firstLineIndent = -self.BASE_INDENT / 2
            
            if properties['list_style'] == 'bullet':
                style.bulletText = self.BULLET_CHARS[list_level - 1] if list_level > 0 else None
            elif properties['list_style'] == 'numbered':
                style.bulletText = f"{list_level}."
        
        return style
    
    def render_element(self, element: DocumentElement, canvas_obj: canvas.Canvas, 
                      page_height: float) -> None:
        """Renders a single element according to its blueprint specifications."""
        element_type = element['type']
        bbox = element['bbox']
        content = element['content']
        
        # Convert normalized coordinates to absolute coordinates
        x0 = bbox[0] * self.page_width
        y0 = bbox[1] * self.page_height
        x1 = bbox[2] * self.page_width
        y1 = bbox[3] * self.page_height
        
        width = x1 - x0
        height = y1 - y0
        
        if element_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3', 'list_item']:
            # Start with original style
            style = self.create_paragraph_style(element)
            p = Paragraph(content, style)
            
            # Check initial fit
            actual_width, actual_height = p.wrap(width, height)
            
            # Dynamic font size adjustment loop
            original_font_size = style.fontSize
            style_to_use = style
            
            while actual_height > height and style_to_use.fontSize > 6:  # Don't go smaller than 6pt font
                new_font_size = style_to_use.fontSize - 1
                print(f"-> Content overflow detected for {element['id']}. Reducing font size from {style_to_use.fontSize}pt to {new_font_size}pt.")
                
                # Create a new style with the smaller font
                style_to_use = ParagraphStyle(
                    name=f"style_{element['id']}_resized",
                    parent=style,  # Inherit all other properties
                    fontSize=new_font_size,
                    leading=new_font_size * 1.2  # Adjust leading as well
                )
                
                # Recreate paragraph with new style
                p = Paragraph(content, style_to_use)
                actual_width, actual_height = p.wrap(width, height)
            
            # Final check - if still doesn't fit, log warning but proceed
            if actual_height > height:
                print(f"⚠️  Warning: Element {element['id']} content still overflows after font reduction. "
                      f"Required height: {actual_height:.2f}, Available: {height:.2f}")

            # Draw the paragraph at its precise location
            p.drawOn(canvas_obj, x0, page_height - y1)
        
        elif element_type == 'table':
            # Enhanced table rendering with basic structure
            self.render_table_element(element, canvas_obj, x0, y0, x1, y1, page_height)
        
        elif element_type == 'image':
            # Enhanced image rendering placeholder
            self.render_image_element(element, canvas_obj, x0, y0, x1, y1, page_height)
        
        elif element_type == 'caption':
            # Render caption with smaller font
            style = ParagraphStyle(
                name=f"caption_{element['id']}",
                fontName="Helvetica",
                fontSize=8,
                alignment=TA_CENTER,
                textColor=(0.5, 0.5, 0.5)
            )
            p = Paragraph(content, style)
            p.wrap(width, height)
            p.drawOn(canvas_obj, x0, page_height - y1)
        
        elif element_type in ['header', 'footer']:
            # Render header/footer
            style = ParagraphStyle(
                name=f"{element_type}_{element['id']}",
                fontName="Helvetica",
                fontSize=10,
                alignment=TA_CENTER,
                textColor=(0.3, 0.3, 0.3)
            )
            p = Paragraph(content, style)
            p.wrap(width, height)
            p.drawOn(canvas_obj, x0, page_height - y1)
        
        # Create bookmark for headings (for TOC generation)
        if element_type in ['heading_1', 'heading_2', 'heading_3']:
            canvas_obj.bookmarkPage(key=element['id'])
            self.element_page_map[element['id']] = self.current_page
    
    def render_table_element(self, element: DocumentElement, canvas_obj: canvas.Canvas, 
                           x0: float, y0: float, x1: float, y1: float, page_height: float) -> None:
        """Enhanced table rendering with basic structure."""
        width = x1 - x0
        height = y1 - y0
        
        # Draw table border
        canvas_obj.setStrokeColor(black)
        canvas_obj.setLineWidth(1)
        canvas_obj.rect(x0, page_height - y1, width, height)
        
        # Draw table content
        canvas_obj.setFont("Helvetica", 10)
        canvas_obj.setFillColor(black)
        
        # Split content into lines and render
        lines = element['content'].split('\n')
        line_height = 12
        y_pos = page_height - y1 + 10  # Start 10 points from top
        
        for i, line in enumerate(lines):
            if y_pos > page_height - y0 - line_height:  # Check if we're still within bounds
                break
            canvas_obj.drawString(x0 + 5, y_pos, line[:50])  # Limit line length
            y_pos += line_height
    
    def render_image_element(self, element: DocumentElement, canvas_obj: canvas.Canvas, 
                           x0: float, y0: float, x1: float, y1: float, page_height: float) -> None:
        """Enhanced image rendering with placeholder and metadata."""
        width = x1 - x0
        height = y1 - y0
        
        # Draw image placeholder border
        canvas_obj.setStrokeColor((0.7, 0.7, 0.7))
        canvas_obj.setLineWidth(2)
        canvas_obj.setDash(5, 5)  # Dashed border
        canvas_obj.rect(x0, page_height - y1, width, height)
        canvas_obj.setDash(1, 0)  # Reset to solid
        
        # Draw image placeholder text
        canvas_obj.setFont("Helvetica", 12)
        canvas_obj.setFillColor((0.5, 0.5, 0.5))
        
        # Center the text in the image area
        text = f"[IMAGE: {element['content'][:30]}...]"
        text_width = canvas_obj.stringWidth(text, "Helvetica", 12)
        text_x = x0 + (width - text_width) / 2
        text_y = page_height - y1 + height / 2
        
        canvas_obj.drawString(text_x, text_y, text)
    
    def reconstruct_document(self, output_path: str) -> Dict[str, int]:
        """
        Reconstructs the document from blueprint to PDF using canvas-based positioning.
        Returns element_page_map for TOC generation.
        """
        # Create canvas for precise positioning
        c = canvas.Canvas(output_path, pagesize=(self.page_width, self.page_height))
        
        for page_data in self.blueprint['pages']:
            page_num = page_data['page_number']
            page_height = float(page_data['dimensions'][1])
            
            print(f"Rendering page {page_num} with {len(page_data['elements'])} elements")
            
            # Get reading order for this page
            reading_order = page_data['reading_order']
            elements = {el['id']: el for el in page_data['elements']}
            
            # Render elements in reading order using their exact bbox positions
            for element_id in reading_order:
                if element_id in elements:
                    element = elements[element_id]
                    self.render_element(element, c, page_height)
            
            # Create bookmark for headings (for TOC generation)
            for element in page_data['elements']:
                if element['type'] in ['heading_1', 'heading_2', 'heading_3']:
                    c.bookmarkPage(key=element['id'])
                    self.element_page_map[element['id']] = page_num
            
            # Add new page if not the last page
            if page_num < len(self.blueprint['pages']):
                c.showPage()
            
            self.current_page = page_num + 1
        
        # Save the PDF
        c.save()
        
        print(f"Document reconstructed with canvas-based positioning: {output_path}")
        return self.element_page_map


def reconstruct_from_blueprint(
    blueprint_path: str,
    output_path: str = "reconstructed_document.pdf"
) -> Dict[str, int]:
    """
    Main function to reconstruct a document from a blueprint.
    Returns element_page_map for TOC generation.
    """
    # Load blueprint
    blueprint = load_blueprint(blueprint_path)
    if not blueprint:
        print(f"Failed to load blueprint: {blueprint_path}")
        return {}
    
    # Create reconstructor and build document
    reconstructor = DocumentReconstructor(blueprint)
    element_page_map = reconstructor.reconstruct_document(output_path)
    
    return element_page_map


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_reconstructor.py <translated_blueprint.json> [output.pdf]")
        sys.exit(1)
    
    blueprint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "reconstructed_document.pdf"
    
    element_page_map = reconstruct_from_blueprint(blueprint_path, output_path)
    print(f"Element page map: {element_page_map}") 