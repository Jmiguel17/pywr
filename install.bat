@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM Pywr editable install on Windows
REM Uses a stable Python 3.12 conda env and initialises MSVC if installed.
REM ============================================================================

set "ENV_NAME=pywr312"
set "PYTHON_VERSION=3.12"

REM pywr source directory; defaults to current dir if not passed as %1
if "%~1"=="" (
  set "PYWR_SRC=%cd%"
) else (
  set "PYWR_SRC=%~1"
)

echo [INFO] pywr source: "%PYWR_SRC%"
echo [INFO] Target env: "%ENV_NAME%"

REM ---- Find Conda ------------------------------------------------------------
where conda >nul 2>nul || (
  echo [ERROR] 'conda' not found in PATH.
  echo         Open Anaconda Prompt and re-run this script.
  exit /b 1
)

REM ---- Create env if missing ------------------------------------------------
conda env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Conda env "%ENV_NAME%" not found. Creating it now...

  call conda create -y -n "%ENV_NAME%" -c conda-forge ^
    python=%PYTHON_VERSION% ^
    cython ^
    numpy ^
    scipy ^
    pandas ^
    networkx ^
    packaging ^
    setuptools ^
    wheel ^
    pip ^
    glpk ^
    lpsolve55 ^
    openpyxl ^
    pytables

  if errorlevel 1 (
    echo [ERROR] Failed to create Conda env "%ENV_NAME%".
    exit /b 1
  )
)

REM ---- Activate target env --------------------------------------------------
call conda activate "%ENV_NAME%"
if errorlevel 1 (
  echo [ERROR] Could not activate "%ENV_NAME%".
  exit /b 1
)

echo [INFO] Using env: "%CONDA_PREFIX%"

set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found at "%PYTHON_EXE%".
  exit /b 1
)

echo [INFO] Python: "%PYTHON_EXE%"
"%PYTHON_EXE%" -m pip --version

REM ---- Enforce Python <= 3.13 ------------------------------------------------
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] <= (3,13) else 1)"
if errorlevel 1 (
  echo [ERROR] This script does not support Python 3.14 for Pywr source builds.
  echo         Use Python 3.12 or 3.13.
  echo         Current env: "%CONDA_PREFIX%"
  exit /b 1
)

REM ---- Check Python build/runtime dependencies -------------------------------
"%PYTHON_EXE%" -c "import Cython; print('[INFO] Cython:', Cython.__version__)"
if errorlevel 1 (
  echo [ERROR] Cython is missing.
  exit /b 1
)

"%PYTHON_EXE%" -c "import numpy; print('[INFO] NumPy:', numpy.__version__)"
if errorlevel 1 (
  echo [ERROR] NumPy is missing.
  exit /b 1
)

"%PYTHON_EXE%" -c "import pandas, scipy, networkx, tables, openpyxl; print('[INFO] Main Python dependencies found')"
if errorlevel 1 (
  echo [ERROR] One or more Python dependencies are missing.
  echo         Try:
  echo         conda install -n %ENV_NAME% -c conda-forge pandas scipy networkx pytables openpyxl
  exit /b 1
)

REM ---- Point compiler/linker to Conda headers/libs ---------------------------
set "INCLUDE=%CONDA_PREFIX%\Library\include;%INCLUDE%"
set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"
set "CL=/I%CONDA_PREFIX%\Library\include %CL%"
set "LINK=/LIBPATH:%CONDA_PREFIX%\Library\lib %LINK%"
set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"

REM ---- Check GLPK ------------------------------------------------------------
if not exist "%CONDA_PREFIX%\Library\include\glpk.h" (
  echo [ERROR] glpk.h not found under "%CONDA_PREFIX%\Library\include".
  echo         Try:
  echo         conda install -n %ENV_NAME% -c conda-forge glpk
  exit /b 1
)

echo [INFO] Found GLPK header: "%CONDA_PREFIX%\Library\include\glpk.h"

REM ---- Check lpsolve ---------------------------------------------------------
if exist "%CONDA_PREFIX%\Library\include\lpsolve\lp_lib.h" (
  echo [INFO] Found lpsolve header: "%CONDA_PREFIX%\Library\include\lpsolve\lp_lib.h"
) else if exist "%CONDA_PREFIX%\Library\include\lp_lib.h" (
  echo [INFO] Found lpsolve header: "%CONDA_PREFIX%\Library\include\lp_lib.h"
) else (
  echo [WARN] lp_lib.h not found. Continuing with GLPK available.
)

REM ---- Initialise MSVC -------------------------------------------------------
call :InitMSVC
if errorlevel 1 (
  exit /b 1
)

REM ---- Install Pywr editable ------------------------------------------------
pushd "%PYWR_SRC%"
if errorlevel 1 (
  echo [ERROR] Could not enter pywr source directory "%PYWR_SRC%".
  exit /b 1
)

echo [INFO] Installing pywr from "%cd%"...

REM Do not upgrade pip inside this script.
REM --no-build-isolation uses the Conda-installed Cython/NumPy/GLPK stack.
REM --no-deps avoids pip trying to resolve Conda packages again.

"%PYTHON_EXE%" -m pip install --no-cache-dir --no-build-isolation --no-deps -v -e .
set "RC=%ERRORLEVEL%"

popd

if not "%RC%"=="0" (
  echo [ERROR] pywr install failed. See errors above.
  exit /b %RC%
)

REM ---- Verification ----------------------------------------------------------
"%PYTHON_EXE%" -c "import pywr; print('[OK] pywr version:', getattr(pywr, '__version__', '?'))"
"%PYTHON_EXE%" -c "import importlib.util as u; print('[OK] GLPK extension found:', bool(u.find_spec('pywr.solvers.cython_glpk')))"
"%PYTHON_EXE%" -c "import importlib.util as u; print('[OK] lpsolve solver found:', bool(u.find_spec('pywr.solvers.lpsolve')))"

echo [SUCCESS] pywr installed in "%ENV_NAME%".
endlocal
exit /b 0


REM ============================================================================
REM Subroutine: initialise MSVC
REM ============================================================================
:InitMSVC

echo [INFO] Checking for MSVC compiler...

where cl.exe >nul 2>nul
if "%ERRORLEVEL%"=="0" (
  echo [INFO] MSVC compiler already available:
  where cl.exe
  exit /b 0
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%VSWHERE%" (
  echo [ERROR] MSVC compiler not found.
  echo.
  echo Install Microsoft C++ Build Tools:
  echo   1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  echo   2. Install workload: Desktop development with C++
  echo   3. Include: MSVC v143 x64/x86 build tools
  echo   4. Include: Windows 10 or Windows 11 SDK
  echo   5. Reopen Anaconda Prompt
  echo   6. Run install.bat again
  echo.
  exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
  set "VSINSTALL=%%i"
)

if not defined VSINSTALL (
  echo [ERROR] Visual Studio C++ Build Tools were not found.
  echo.
  echo Install Microsoft C++ Build Tools:
  echo   1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  echo   2. Install workload: Desktop development with C++
  echo   3. Include: MSVC v143 x64/x86 build tools
  echo   4. Include: Windows 10 or Windows 11 SDK
  echo   5. Reopen Anaconda Prompt
  echo   6. Run install.bat again
  echo.
  exit /b 1
)

set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"

if not exist "%VSDEVCMD%" (
  echo [ERROR] VsDevCmd.bat not found:
  echo         "%VSDEVCMD%"
  echo.
  echo Your Visual Studio installation is incomplete.
  echo Install the "Desktop development with C++" workload.
  exit /b 1
)

echo [INFO] Initialising MSVC from:
echo        "%VSDEVCMD%"

call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 (
  echo [ERROR] Failed to initialise MSVC environment.
  exit /b 1
)

where cl.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] MSVC initialisation completed, but cl.exe is still unavailable.
  exit /b 1
)

echo [INFO] MSVC compiler found:
where cl.exe

exit /b 0