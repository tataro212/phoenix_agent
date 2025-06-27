# Phoenix Agent v2: Document Translation Pipeline

## Overview

The Phoenix Agent is a transparent, sequential Assembly Line system for document analysis and translation. It follows the core philosophy of **"The Right Tool for the Right Job"** with absolute modularity and verifiability at every step.

## Core Architecture

### Single Source of Truth: Document Blueprint
- A structured JSON object (`DocumentBlueprint`) is the lifeblood of the system
- Passed between stations on the assembly line with no side channels
- Strictly validated schema using Python TypedDict

### Four-Station Assembly Line

#### Station 1: The Surveyor (`document_parser.py`)
**Job:** Creates flawless `document_blueprint.json` from source PDF

**Tools Used:**
- **YOLO (DocLayNet):** Identifies location and class of layout blocks
- **Nougat:** High-fidelity OCR and structure extraction within blocks
- **PyMuPDF:** Formatting property extraction (font, alignment, etc.)

**Process:**
1. **Pass 1:** High-level layout segmentation (YOLO at 300 DPI)
2. **Pass 2:** Detailed content extraction (Nougat + PyMuPDF)
3. **Pass 3:** Reading order determination
4. **Verification:** Visual debugger creates colored, labeled boxes

#### Station 2: The Diplomat (`translation_service.py`)
**Job:** Translates blueprint content with context-aware chunking

**Core Tactic:** Isolate Language from Structure
- LLM only translates strings, never generates formatting
- Structure preserved in Blueprint's type and properties fields

**Context-Aware Translation:**
- **Semantic Context:** Preceding/following sentences
- **Structural Context:** Element type metadata
- **Structured Prompt:** XML-like format for clear role demarcation

#### Station 3: The Architect (`document_reconstructor.py`)
**Job:** Transforms translated blueprint into concrete PDF

**Blueprint-Driven Engineering:**
- Every visual aspect dictated by Blueprint properties
- Direct mapping from properties to ReportLab parameters
- Perfect structure preservation through 1:1 property translation

#### Station 4: The Librarian (`toc_generator.py`)
**Job:** Generates pixel-perfect, interactive Table of Contents

**Two-Pass Compilation:**
- **Pass 1:** Document compilation with bookmark creation
- **Pass 2:** TOC generation with hyperlinks and dot leaders

## Installation

```bash
# Install required dependencies
pip install reportlab pymupdf opencv-python pillow numpy

# For YOLO integration (when ready)
pip install ultralytics

# For Nougat integration (when ready)
pip install nougat-ocr
```

## Usage

<<<<<<< HEAD
### 🎯 Interactive Mode (Recommended)

The easiest way to use Phoenix Agent is through the interactive mode with file dialogs:

#### Option 1: Double-click launcher (Windows)
```bash
# Simply double-click this file:
run_interactive.bat
```

#### Option 2: Python launcher
```bash
# Run the interactive launcher
python run_interactive.py
```

#### Option 3: Command line interactive mode
```bash
# Use the --interactive flag
python phoenix_orchestrator.py --interactive
```

The interactive mode will:
1. **Open a file dialog** to select your input PDF file
2. **Open a folder dialog** to choose where to save the translation
3. **Show a language selection dialog** to pick the target language
4. **Confirm your choices** before starting the translation
5. **Run the complete pipeline** automatically

### Command Line Mode

=======
>>>>>>> ef046a9256cf34a0f298c6a336a3eb4d3599a7d9
### Full Pipeline
```bash
python phoenix_orchestrator.py input.pdf --target-language en --output-dir phoenix_output
```

### Individual Stations
```bash
# Station 1: Surveyor
python phoenix_orchestrator.py input.pdf --station 1

# Station 2: Diplomat
python phoenix_orchestrator.py input.pdf --station 2 --target-language es

# Station 3: Architect
python phoenix_orchestrator.py input.pdf --station 3

# Station 4: Librarian
python phoenix_orchestrator.py input.pdf --station 4
```

### Advanced Options
```bash
# With visual verification
python phoenix_orchestrator.py input.pdf --page-images page_images_dir

# Skip verification
python phoenix_orchestrator.py input.pdf --skip-verification

# Custom output directory
python phoenix_orchestrator.py input.pdf --output-dir my_output
```

## File Structure

```
phoenix_agent_v2/
├── document_blueprint.py      # Single source of truth schema
├── document_parser.py         # Station 1: Surveyor
├── visual_debugger.py         # Verification tool
├── translation_service.py     # Station 2: Diplomat
├── document_reconstructor.py  # Station 3: Architect
├── toc_generator.py          # Station 4: Librarian
├── phoenix_orchestrator.py   # Main orchestrator
<<<<<<< HEAD
├── run_interactive.py        # Interactive mode launcher
├── run_interactive.bat       # Windows batch launcher
=======
>>>>>>> ef046a9256cf34a0f298c6a336a3eb4d3599a7d9
└── README.md                 # This file
```

## Output Files

After running the pipeline, you'll find:

```
phoenix_output/
├── document_blueprint.json           # Original blueprint
├── translated_blueprint.json         # Translated blueprint
├── reconstructed_document.pdf        # Final PDF
├── table_of_contents.pdf            # Interactive TOC
├── element_page_map.json            # Page mapping for TOC
└── debug_output/                    # Visual verification images
    ├── debug_page_001.png
    ├── debug_page_002.png
    ├── element_legend.png
    └── debug_summary.txt
```

## Implementation Details

### Document Blueprint Schema
```python
class DocumentBlueprint(TypedDict):
    metadata: DocumentMetadata
    pages: List[Page]

class DocumentElement(TypedDict):
    id: str
    type: ElementType
    bbox: Tuple[float, float, float, float]
    content: str
    properties: ElementProperties
    children: List[DocumentElement]
```

### Translation Prompt Template
```xml
<System>
You are an expert academic translator...
**RULES:**
1. Translate ONLY the content within the <TRANSLATE_THIS> tag.
2. Do NOT translate the content in the <CONTEXT_...> tags.
3. The number of paragraphs MUST exactly match...
4. Do NOT add any formatting, markdown, or commentary.
</System>

<ContextualInformation>
    <PrecedingText>...</PrecedingText>
    <FollowingText>...</FollowingText>
    <StructuralHints>...</StructuralHints>
</ContextualInformation>

<TRANSLATE_THIS>
Content to translate...
</TRANSLATE_THIS>
```

### Element Rendering Engine
```python
# Direct property mapping from blueprint to ReportLab
style = ParagraphStyle(
    fontName=element['properties']['font_name'],
    fontSize=element['properties']['font_size'],
    alignment=alignment_map[element['properties']['alignment']],
    firstLineIndent=element['properties']['indentation']
)
```

## Integration Points

### YOLO Integration
Replace `DummyYOLOModel` in `document_parser.py`:
```python
from ultralytics import YOLO
yolo_model = YOLO('path/to/your/trained/model.pt')
```

### Nougat Integration
Replace `DummyNougatModel` in `document_parser.py`:
```python
from nougat import NougatModel
nougat_model = NougatModel.from_pretrained("facebook/nougat-base")
```

### Gemini API Integration
Replace `DummyGeminiAPI` in `translation_service.py`:
```python
import google.generativeai as genai
genai.configure(api_key='your_api_key')
model = genai.GenerativeModel('gemini-pro')
```

## Verification and Quality Assurance

### Visual Debugger
- Creates colored, labeled bounding boxes for each element
- Generates summary report with element statistics
- Provides element type legend
- **Non-negotiable step** before proceeding to translation

### Validation
- Schema validation at every step
- Blueprint integrity checks
- Translation chunk validation
- Page mapping verification

## Error Handling

The system includes comprehensive error handling:
- Input validation
- Model loading failures
- Translation errors
- PDF generation issues
- Graceful degradation with warnings

## Performance Considerations

- **Parallel Processing:** Each station can run independently
- **Caching:** Translation results can be cached
- **Memory Management:** Large documents processed page by page
- **GPU Acceleration:** YOLO and Nougat can use GPU when available

## Future Enhancements

- **Real-time Processing:** Stream processing for large documents
- **Batch Processing:** Multiple document pipeline
- **Custom Models:** Fine-tuned models for specific domains
- **API Service:** REST API for cloud deployment
- **GUI Interface:** Web-based user interface

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Loading Errors**
   - Ensure YOLO model path is correct
   - Check Nougat model availability
   - Verify API keys for Gemini

3. **Memory Issues**
   - Process documents page by page
   - Use smaller batch sizes
   - Enable GPU acceleration

4. **Translation Quality**
   - Adjust chunking parameters
   - Fine-tune prompt templates
   - Use domain-specific models

## Contributing

The Phoenix Agent follows strict architectural principles:
- **Single Responsibility:** Each module has one job
- **Transparency:** All data flows are explicit
- **Verifiability:** Every step produces inspectable artifacts
- **Modularity:** Components can be replaced independently

## License

This project follows the same license as the parent repository.

---

**The Phoenix Agent: Where transparency meets precision in document translation.** 