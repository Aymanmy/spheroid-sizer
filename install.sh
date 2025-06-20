#!/bin/bash
# Spheroid Sizer Installation Script
# Usage: ./install.sh [--full|--dev|--minimal]

set -e  # Exit on error

# ===== Configuration =====
PYTHON=python3  # Change to 'python' if needed
VENV_NAME=".venv"  # Virtual environment folder
REQUIREMENTS_CORE="requirements.txt"
REQUIREMENTS_DEV="requirements-dev.txt"
REQUIREMENTS_OPT="requirements-optional.txt"

# ===== Functions =====
activate_venv() {
    if [ -d "$VENV_NAME" ]; then
        echo "Activating virtual environment..."
        if [ -f "$VENV_NAME/bin/activate" ]; then
            source "$VENV_NAME/bin/activate"  # Linux/Mac
        else
            source "$VENV_NAME/Scripts/activate"  # Windows
        fi
    fi
}

install_deps() {
    echo "Installing $1 dependencies..."
    $PYTHON -m pip install --upgrade pip
    case "$1" in
        --full)
            pip install -r $REQUIREMENTS_CORE
            pip install -r $REQUIREMENTS_OPT
            ;;
        --dev)
            pip install -r $REQUIREMENTS_CORE
            pip install -r $REQUIREMENTS_DEV
            pip install -r $REQUIREMENTS_OPT
            ;;
        *)
            pip install -r $REQUIREMENTS_CORE  # Minimal install
            ;;
    esac
}

# ===== Main =====
echo "=== Spheroid Sizer Installer ==="

# Create virtual environment (optional)
read -p "Create virtual environment? [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    $PYTHON -m venv $VENV_NAME
    activate_venv
fi

# Select installation mode
if [ "$1" == "--full" ]; then
    install_deps --full
elif [ "$1" == "--dev" ]; then
    install_deps --dev
else
    echo "Select installation mode:"
    echo "1) Minimal (core only)"
    echo "2) Full (core + optional features)"
    echo "3) Developer (core + optional + dev tools)"
    read -p "Choice [1-3]: " choice

    case $choice in
        2) install_deps --full ;;
        3) install_deps --dev ;;
        *) install_deps --minimal ;;
    esac
fi

# Verify OpenCV installation
echo "Verifying OpenCV..."
$PYTHON -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo "=== Installation complete ==="
echo "Run the app with: $PYTHON src/spheroid_sizer.py"