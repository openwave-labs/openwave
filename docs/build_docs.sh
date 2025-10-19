#!/bin/bash

# Build script for OpenWave documentation
# Generates comprehensive HTML documentation with all features

set -e  # Exit on error

echo "======================================"
echo "OpenWave Documentation Builder"
echo "======================================"

# Change to docs directory
cd "$(dirname "$0")"

# Install documentation dependencies
echo "Installing documentation dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p _static _templates api/generated

# Generate custom CSS
echo "Creating custom styles..."
cat > _static/custom.css << 'EOF'
/* Custom styles for OpenWave documentation */
.wy-nav-content {
    max-width: 1200px;
}

.rst-content code {
    font-size: 90%;
}

.rst-content .section {
    margin-bottom: 30px;
}

/* Improve readability of inheritance diagrams */
.inheritance svg {
    max-width: 100%;
    height: auto;
}

/* Style for dependency graphs */
.graphviz svg {
    max-width: 100%;
    height: auto;
    border: 1px solid #e1e4e5;
    border-radius: 4px;
    padding: 10px;
    background: #f8f9fa;
}

/* Better code block styling */
.highlight pre {
    padding: 12px;
    border-radius: 4px;
}

/* Module index styling */
.modindex-jumpbox {
    margin: 20px 0;
    padding: 10px;
    background: #f0f0f0;
    border-radius: 4px;
}
EOF

# Clean old API documentation
echo "Cleaning old API documentation..."
rm -rf api/*.rst

# Regenerate API documentation from source
echo "Generating API documentation from source code..."
python -m sphinx.ext.apidoc -f -e -M -o api ../openwave

# Generate dependency graphs
echo "Generating dependency graphs..."
python generate_deps.py || echo "Warning: Some dependency graphs could not be generated"

# Build the documentation
echo "Building HTML documentation..."
make clean
make html

# Check if build was successful
if [ -d "_build/html" ]; then
    echo ""
    echo "======================================"
    echo "Documentation built successfully!"
    echo "======================================"
    echo ""
    echo "View documentation at:"
    echo "  file://$(pwd)/_build/html/index.html"
    echo ""
    echo "Or start a local server:"
    echo "  python -m http.server 8000 --directory _build/html"
    echo ""
    echo "Then visit: http://localhost:8000"
    echo ""
    echo "For live reload during development:"
    echo "  make watch"
else
    echo "Error: Documentation build failed"
    exit 1
fi