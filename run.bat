@ECHO off
setlocal
REM This script is used for activating conda environment and creating
ECHO Searching for Anaconda/miniconda installation and environment for this project

REM Define the name of environment and script
SET ENV_NAME=audio
SET SCRIPT_NAME=main.py
SET LOGS_DIR=logs
SET SERVER_LOG=%LOGS_DIR%\server.log
SET CLIENT_LOG=%LOGS_DIR%\client.log

REM Attempt to find Anaconda or miniconda installation
FOR %%d IN (C D E F G H I J K L M N O P Q R S T U V W X Y Z) DO (
    IF EXIST "%%d:\Users\%USERNAME%\Anaconda3\Scripts\activate.bat" (
        SET ANACONDA_PATH=%%d:\Users\%USERNAME%\Anaconda3
        GOTO Found
    )
    IF EXIST "%%d:\Users\%USERNAME%\miniconda3\Scripts\activate.bat" (
        SET ANACONDA_PATH=%%d:\Users\%USERNAME%\miniconda3
        GOTO Found
    )
)
ECHO.
ECHO No Anaconda/miniconda installation was found on this system under the default installation paths:
ECHO - "%%d:\Users\%USERNAME%\Anaconda3"
ECHO - "%%d:\Users\%USERNAME%\miniconda3"
ECHO Please ensure Anaconda/miniconda is installed or modify the script to reflect its install location
ECHO Anaconda/miniconda not found, please ensure Anaconda/miniconda is installed or modify the script to reflect its install location
GOTO END

:Found
ECHO Anaconda/miniconda installation found at %ANACONDA_PATH%
CALL "%ANACONDA_PATH%\Scripts\activate.bat" "%ANACONDA_PATH%"
ECHO.
ECHO Trying to activate the environment [ %ENV_NAME% ]
REM Check if the environment exists
IF EXIST "%ANACONDA_PATH%\envs\%ENV_NAME%" (
    REM Activate the environment
    GOTO ActivateEnv
) ELSE (
    ECHO There is no environment named %ENV_NAME% in the Anaconda/miniconda installation at %ANACONDA_PATH%
    ECHO Trying to Setup the environment with projects environment.yml
    ECHO.
    REM Create the environment from the environment.yml file
    GOTO CreateEnv
)
ECHO.

:CreateEnv
CALL conda env create -f environment.yml
CALL conda activate %ENV_NAME%
ECHO Environment [ %ENV_NAME% ] successfully created and activated
ECHO.
GOTO Run

:ActivateEnv
CALL conda activate %ENV_NAME%
ECHO Environment [ %ENV_NAME% ] successfully activated
ECHO.
GOTO Run

:Run
REM Check if "logs" directory exists, if not create it
IF NOT EXIST "%LOGS_DIR%" (
    ECHO Creating "logs" directory
    MKDIR "%LOGS_DIR%"
)
REM Check if "server.log" exists (inside logs), if not create it
IF NOT EXIST "%SERVER_LOG%" (
    ECHO Creating "server.log"
    ECHO Server Log File Created on %date% %time% > "%SERVER_LOG%"
)
REM same for "client.log"
IF NOT EXIST "%CLIENT_LOG%" (
    ECHO Creating "client.log"
    ECHO Client Log File Created on %date% %time% > "%CLIENT_LOG%"
)
ECHO Starting program (This might take a few seconds)
python "%~dp0%SCRIPT_NAME%"
PAUSE
CALL conda deactivate
GOTO END

:END
endlocal
PAUSE