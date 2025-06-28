# Phoenix Agent - Clean Environment Setup Script
# This script creates a fresh conda environment with pinned dependencies

Write-Host "🚀 Phoenix Agent - Clean Environment Setup" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if conda is available
try {
    conda --version | Out-Null
    Write-Host "✅ Conda is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Conda is not available. Please install Anaconda or Miniconda first." -ForegroundColor Red
    exit 1
}

# Deactivate any active environment
Write-Host "📋 Deactivating current environment..." -ForegroundColor Yellow
conda deactivate

# Create a completely new, clean environment
Write-Host "🔧 Creating new conda environment 'phoenix_final'..." -ForegroundColor Yellow
conda create --name phoenix_final python=3.10 -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create conda environment" -ForegroundColor Red
    exit 1
}

# Activate the new environment
Write-Host "🔧 Activating 'phoenix_final' environment..." -ForegroundColor Yellow
conda activate phoenix_final

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate conda environment" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies from requirements.txt
Write-Host "📦 Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Write-Host "💡 Try installing dependencies manually:" -ForegroundColor Yellow
    Write-Host "   pip install torch==2.1.0 torchvision==0.16.0 timm==0.6.13" -ForegroundColor Cyan
    Write-Host "   pip install pydantic<2.0 albumentations==1.3.1 nougat-ocr" -ForegroundColor Cyan
    Write-Host "   pip install ultralytics PyMuPDF reportlab python-docx" -ForegroundColor Cyan
    Write-Host "   pip install numpy<2.0.0 opencv-python Pillow google-generativeai" -ForegroundColor Cyan
    exit 1
}

Write-Host "" -ForegroundColor White
Write-Host "🎉 Environment setup completed successfully!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "✅ New environment: phoenix_final" -ForegroundColor Green
Write-Host "✅ Python version: 3.10" -ForegroundColor Green
Write-Host "✅ All dependencies installed with pinned versions" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate the environment: conda activate phoenix_final" -ForegroundColor Cyan
Write-Host "2. Test the pipeline: python phoenix_orchestrator.py --interactive" -ForegroundColor Cyan
Write-Host "3. Or run with a specific PDF: python phoenix_orchestrator.py your_document.pdf" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White 