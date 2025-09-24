#!/bin/bash
# Setup script for SimEx development environment

echo "Setting up SimEx development environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing SimEx in development mode..."
pip install -e .

echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "pytest"
echo ""
echo "To run examples:"
echo "python examples/simex_run.py"
echo "or: simex-run"
