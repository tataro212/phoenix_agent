# Phoenix Agent v2 - Document Translation Pipeline

A sophisticated document translation system that preserves layout fidelity while translating content across languages. The Phoenix Agent uses an assembly-line approach with four specialized stations to process documents from PDF input to translated output.

## 🚀 New Features & Fixes (Latest Update)

### ✅ **Dynamic Font Sizing**
- **Problem Solved**: Content overflow in reconstructed PDFs
- **Solution**: Intelligent font size reduction until text fits within bounding boxes
- **Result**: Clean, professional-looking output without text cut-offs

### ✅ **Word Document Export**
- **New Feature**: Export translated documents to Microsoft Word (.docx) format
- **Benefits**: Easy editing, collaboration, and further formatting
- **Integration**: Automatically generated alongside PDF output

### ✅ **Enhanced Layout Preservation**
- **Improved**: Table and image rendering with proper bounding box positioning
- **Enhanced**: Better handling of complex document structures
- **Result**: Higher fidelity to original document layout

### ✅ **Fixed Nougat Compatibility**
- **Problem**: Dependency conflicts causing `ImageNetInfo` errors
- **Solution**: Pinned dependency versions in `requirements.txt`
- **Result**: Reliable OCR with superior text extraction quality

## 🏗️ Architecture Overview

The Phoenix Agent operates as a four-station assembly line:

### Station 1: The Surveyor 📋
- **Purpose**: Analyzes PDF structure and extracts document blueprint
- **Technology**: Nougat OCR + PyMuPDF fallback
- **Output**: `document_blueprint.json` with spatial coordinates

### Station 2: The Diplomat 🌍
- **Purpose**: Translates content while preserving structure
- **Technology**: Google Gemini API with context-aware chunking
- **Output**: `translated_blueprint.json`

### Station 3: The Architect 🏛️
- **Purpose**: Reconstructs document with layout fidelity
- **Technology**: ReportLab with dynamic font sizing
- **Output**: `reconstructed_document.pdf`

### Station 4: The Librarian 📚
- **Purpose**: Generates interactive Table of Contents
- **Technology**: ReportLab with hyperlinks
- **Output**: `table_of_contents.pdf`

### Station 5: The Scribe ✍️ (NEW)
- **Purpose**: Exports to Microsoft Word format
- **Technology**: python-docx with proper formatting
- **Output**: `translated_document.docx`

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10
- Anaconda or Miniconda
- Google Gemini API key (optional, uses dummy translator if not provided)

### Quick Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd phoenix_agent_v2
   ```

2. **Run the automated setup script**:
   ```bash
   # Windows (PowerShell)
   .\setup_clean_env.ps1
   
   # Or manually:
   conda create --name phoenix_final python=3.10 -y
   conda activate phoenix_final
   pip install -r requirements.txt
   ```

3. **Set up API key** (optional):
   ```bash
   # Windows
   set GEMINI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export GEMINI_API_KEY=your_api_key_here
   ```

### Manual Installation

If the automated setup fails, install dependencies manually:

```bash
# Core ML dependencies (pinned for compatibility)
pip install torch==2.1.0 torchvision==0.16.0 timm==0.6.13

# Nougat + dependencies
pip install pydantic<2.0 albumentations==1.3.1 nougat-ocr

# Application dependencies
pip install ultralytics PyMuPDF reportlab python-docx

# Core Python
pip install numpy<2.0.0 opencv-python Pillow google-generativeai
```

## 🚀 Usage

### Interactive Mode (Recommended)
```bash
python phoenix_orchestrator.py --interactive
```
This opens file dialogs to select:
- Input PDF file
- Output directory
- Target language (defaults to Greek)

### Command Line Mode
```bash
# Basic usage
python phoenix_orchestrator.py input.pdf --target-language el

# With custom output directory
python phoenix_orchestrator.py input.pdf -o my_output -t el

# Skip verification for faster processing
python phoenix_orchestrator.py input.pdf -t el --skip-verification

# Run specific station only
python phoenix_orchestrator.py input.pdf --station 3
```

### Available Options
- `--interactive, -i`: Use file dialogs for input selection
- `--target-language, -t`: Target language code (default: en)
- `--output-dir, -o`: Output directory (default: phoenix_output)
- `--page-images, -p`: Directory with page images for verification
- `--skip-verification, -s`: Skip visual verification step
- `--station`: Run only a specific station (1-5)
- `--nougat-model-path`: Custom Nougat model path

## 📊 Output Files

After successful processing, you'll find these files in your output directory:

- `document_blueprint.json` - Original document structure
- `translated_blueprint.json` - Translated document structure
- `reconstructed_document.pdf` - Final translated PDF
- `table_of_contents.pdf` - Interactive TOC with hyperlinks
- `translated_document.docx` - Word document version
- `element_page_map.json` - Mapping for TOC generation
- `debug_output/` - Visual verification images

## 🧪 Testing

Run the comprehensive test suite to verify all fixes:

```bash
python test_fixes.py
```

This tests:
- ✅ Nougat compatibility
- ✅ Dynamic font sizing
- ✅ Word export functionality
- ✅ Orchestrator integration
- ✅ All dependencies

## 🔧 Troubleshooting

### Nougat Import Errors
If you see `ImageNetInfo` errors:
1. Ensure you're using the pinned versions in `requirements.txt`
2. Rebuild the environment: `conda create --name phoenix_final python=3.10 -y`
3. Install dependencies in order: ML → Nougat → Application

### Content Overflow Issues
The dynamic font sizing should handle most overflow cases automatically. If you still see issues:
- Check the console output for font size reduction messages
- Verify the bounding box coordinates in the blueprint
- Consider adjusting the minimum font size (currently 6pt)

### Word Export Issues
If Word export fails:
- Ensure `python-docx` is installed: `pip install python-docx`
- Check that the translated blueprint is valid
- Verify write permissions in the output directory

## 🎯 Key Improvements

### Layout Fidelity
- **Before**: Text flowed like a simple document
- **After**: Precise positioning using bounding box coordinates
- **Result**: Maintains original document structure

### Content Overflow Handling
- **Before**: Text cut off or warnings ignored
- **After**: Intelligent font size reduction
- **Result**: All content fits properly

### Multiple Output Formats
- **Before**: PDF only
- **After**: PDF + Word + Interactive TOC
- **Result**: Flexible output options for different use cases

### Reliability
- **Before**: Dependency conflicts
- **After**: Pinned versions and fallback mechanisms
- **Result**: Consistent, reproducible results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite: `python test_fixes.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Nougat Team**: For the excellent OCR model
- **Google Gemini**: For the translation API
- **ReportLab**: For PDF generation capabilities
- **PyMuPDF**: For PDF processing and fallback OCR

---

**Phoenix Agent v2** - Transforming documents while preserving their soul. 🚀 