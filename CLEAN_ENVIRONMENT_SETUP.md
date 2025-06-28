# Clean Environment Setup Guide

This guide will help you create a clean environment to resolve the `ImageNetInfo` error and test Station 1 (Nougat) functionality.

## Quick Setup (Automated)

### Option 1: Windows Batch Script
```bash
# Run the batch script
setup_clean_env.bat
```

### Option 2: PowerShell Script
```powershell
# Run the PowerShell script
.\setup_clean_env.ps1
```

## Manual Setup

### Step 1: Create Clean Conda Environment
```bash
# Create new environment with Python 3.10
conda create --name phoenix_prod python=3.10 -y

# Activate the environment
conda activate phoenix_prod
```

### Step 2: Install Requirements
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Test Nougat Functionality
```bash
# Run the Nougat test
python test_nougat_station1.py
```

## Expected Results

If everything is set up correctly, you should see:

```
🚀 Phoenix Agent - Station 1 (Nougat) Test
============================================================
This test verifies that all Nougat dependencies are working
in the clean environment.
============================================================

📋 Running: Nougat Imports
----------------------------------------
🧪 Testing Nougat imports...
   ✅ PyTorch version: 2.1.0
   ✅ TorchVision version: 0.16.0
   ✅ TIMM version: 0.5.4
   ✅ Nougat model import successful
   ✅ Nougat dataset import successful
   ✅ Nougat checkpoint import successful
✅ Nougat Imports PASSED

📋 Running: Document Parser Imports
----------------------------------------
🧪 Testing document parser imports...
   ✅ parse_pdf_to_blueprint import successful
   ✅ YOLOModel import successful
   ✅ NougatProcessor import successful
✅ Document Parser Imports PASSED

📋 Running: Blueprint Creation
----------------------------------------
🧪 Testing blueprint creation...
   ✅ Blueprint creation and validation successful
✅ Blueprint Creation PASSED

📋 Running: Nougat Model Loading
----------------------------------------
🧪 Testing Nougat model loading...
   📥 Loading Nougat model (this may take a moment)...
   ✅ Nougat model loaded successfully
   📊 Model device: cpu
   📊 Model parameters: 1,100,000,000
✅ Nougat Model Loading PASSED

============================================================
📊 Test Results: 4/4 tests passed
🎉 All tests passed! Station 1 should work properly.

💡 Next steps:
   1. Try running the full pipeline with a real PDF
   2. Use: python phoenix_orchestrator.py --interactive
   3. Or: python phoenix_orchestrator.py your_document.pdf
```

## Troubleshooting

### If you see import errors:
1. Ensure you're in the correct environment: `conda activate phoenix_prod`
2. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
3. Upgrade pip: `pip install --upgrade pip`

### If Nougat model download fails:
1. Check your internet connection
2. Try downloading manually: `python -c "from nougat import NougatModel; NougatModel.from_pretrained('facebook/nougat-base')"`

### If you still see ImageNetInfo errors:
1. This indicates you're not in the clean environment
2. Deactivate current environment: `conda deactivate`
3. Remove old environment: `conda env remove --name phoenix_prod`
4. Follow the setup steps again

## Next Steps

Once the test passes successfully:

1. **Test with a real PDF:**
   ```bash
   python phoenix_orchestrator.py --interactive
   ```

2. **Or run specific stations:**
   ```bash
   # Test only Station 1
   python phoenix_orchestrator.py your_document.pdf --station 1
   
   # Run full pipeline
   python phoenix_orchestrator.py your_document.pdf
   ```

3. **Check the output:**
   - Look for `document_blueprint.json` in the `phoenix_output` directory
   - This should now contain high-quality text extraction from Nougat

## Environment Management

- **Activate environment:** `conda activate phoenix_prod`
- **Deactivate environment:** `conda deactivate`
- **List environments:** `conda env list`
- **Remove environment:** `conda env remove --name phoenix_prod`

## Key Benefits of Clean Environment

1. **No dependency conflicts** - Fresh installation of all packages
2. **Correct versions** - All packages installed at compatible versions
3. **No ImageNetInfo errors** - Clean torch/torchvision installation
4. **Proper Nougat functionality** - Full text extraction capabilities 