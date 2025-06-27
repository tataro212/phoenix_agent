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
            self.page_width = first_page['dimensions'][0]
            self.page_height = first_page['dimensions'][1]
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
        
        style = ParagraphStyle(
            name=f"style_{element['id']}",
            fontName=properties['font_name'],
            fontSize=properties['font_size'],
            alignment=alignment_map.get(properties['alignment'], TA_LEFT),
            firstLineIndent=properties['indentation'],
            textColor=properties['text_color'],
            spaceAfter=6,
            spaceBefore=6
        )
        
        # Handle bold and italic
        if properties['font_weight'] == 'bold':
            style.fontName = f"{properties['font_name']}-Bold"
        if properties['is_italic']:
            style.fontName = f"{properties['font_name']}-Italic"
        
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
            # Create paragraph style
            style = self.create_paragraph_style(element)
            
            # Create paragraph
            p = Paragraph(content, style)
            
            # Wrap and draw the paragraph
            wrapped_height = p.wrap(width, height)
            if wrapped_height <= height:
                p.drawOn(canvas_obj, x0, page_height - y1)
            else:
                # Handle overflow by truncating or adjusting
                p.drawOn(canvas_obj, x0, page_height - y1)
        
        elif element_type == 'table':
            # Placeholder for table rendering
            # TODO: Implement table rendering based on children elements
            canvas_obj.setFont("Helvetica", 10)
            canvas_obj.drawString(x0, page_height - y0, f"[TABLE: {content}]")
        
        elif element_type == 'image':
            # Placeholder for image rendering
            canvas_obj.setFont("Helvetica", 10)
            canvas_obj.drawString(x0, page_height - y0, f"[IMAGE: {content}]")
        
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
    
    def reconstruct_document(self, output_path: str) -> Dict[str, int]:
        """
        Reconstructs the document from blueprint to PDF.
        Returns element_page_map for TOC generation.
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=(self.page_width, self.page_height),
            rightMargin=0,
            leftMargin=0,
            topMargin=0,
            bottomMargin=0
        )
        
        story = []
        
        for page_data in self.blueprint['pages']:
            # Create a new page
            if story:
                story.append(PageBreak())
            
            # Get reading order for this page
            reading_order = page_data['reading_order']
            elements = {el['id']: el for el in page_data['elements']}
            
            # Render elements in reading order
            for element_id in reading_order:
                if element_id in elements:
                    element = elements[element_id]
                    
                    # For now, add content to story (simplified approach)
                    # In a full implementation, we'd use canvas for precise positioning
                    if element['type'] in ['paragraph', 'heading_1', 'heading_2', 'heading_3', 'list_item']:
                        style = self.create_paragraph_style(element)
                        p = Paragraph(element['content'], style)
                        story.append(p)
                        story.append(Spacer(1, 6))
            
            self.current_page += 1
        
        # Build the PDF
        doc.build(story)
        
        print(f"Document reconstructed: {output_path}")
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