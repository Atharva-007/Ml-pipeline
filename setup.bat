@echo off
echo Setting up ML Pipeline Project...

:: Create virtual environment
python -m venv venv
echo Virtual environment created.

:: Activate virtual environment
call venv\Scripts\activate.bat
echo Virtual environment activated.

:: Install requirements
pip install -r requirements.txt
echo Requirements installed.

:: Create directories
mkdir data\raw
mkdir data\processed
mkdir models
mkdir plots
mkdir logs
echo Directories created.

echo Setup completed successfully!
pause