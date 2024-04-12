REM creates batch file that allows to start the main processes of the program
REM this additional setup script is needed to allow usage on different systems with either Anaconda or miniconda installed
@echo off
setlocal

REM define the name of environment and script 
set ENV_NAME=audio
set SCRIPT_NAME=main.py

REM attempt to find Anaconda or miniconda installation
for %%d in (C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
    if exist "%%d:\Users\%USERNAME%\Anaconda3\Scripts\activate.bat" (
        set ANACONDA_PATH=%%d:\Users\%USERNAME%\Anaconda3
        goto Found
    )
    if exist "%%d:\Users\%USERNAME%\miniconda3\Scripts\activate.bat" (
        set ANACONDA_PATH=%%d:\Users\%USERNAME%\miniconda3
        goto Found
    )
)

echo Anaconda/miniconda not found, please ensure Anaconda/miniconda is intsalled or modify the script to reflect its install location.
goto END

:Found
echo Found Anaconda/miniconda at %ANACONDA_PATH%

REM write the run script
echo @echo off > run.bat
echo CALL "%ANACONDA_PATH%\Scripts\activate.bat" "%ANACONDA_PATH%" >> run.bat
echo CALL conda activate %ENV_NAME% >> run.bat
echo python "%~dp0%SCRIPT_NAME%" >> run.bat
echo pause >> run.bat
echo CALL conda deactivate >> run.bat
echo exit >> run.bat

echo setup complete. You can now run the program using 'run.bat'.
pause

:END
endlocal
pause