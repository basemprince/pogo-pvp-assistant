@echo off

echo Creating conda environment...
conda env create -f environment.yml

echo Activating environment...
call conda activate pogo-pvp-assistant

echo Installing adb...
winget install --id=Google.AndroidSDK.PlatformTools -e --source=winget

echo Setup complete!
pause
