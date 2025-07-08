@echo off
set ENV_NAME=pogo-pvp-assistant

:: Check if env exists
conda env list | findstr /B /C:"%ENV_NAME%" >nul
if %ERRORLEVEL%==0 (
    echo Environment "%ENV_NAME%" already exists. Removing it...
    conda env remove -n %ENV_NAME% -y
)

echo Creating conda environment...
conda env create -f environment.yml

echo Activating environment...
call conda activate %ENV_NAME%

echo Installing adb...
winget install --id=Google.AndroidSDK.PlatformTools -e --source=winget

echo Setup complete!
pause
