@echo off
REM Setup script for SimEx development environment (Windows)

echo Setting up SimEx development environment...

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install development dependencies
echo Installing development dependencies...
pip install -r requirements-dev.txt

REM Install package in development mode
echo Installing SimEx in development mode...
pip install -e .

echo Setup complete!
echo.
echo To activate the environment in the future, run:
echo .venv\Scripts\activate.bat
echo.
echo To run tests:
echo pytest
echo.
echo To run examples:
echo python examples\simex_run.py
echo or: simex-run
