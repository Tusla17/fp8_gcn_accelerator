@echo off
setlocal enabledelayedexpansion

echo ==================================================================
echo   RUNNING VIVADO SYNTHESIS FOR FP8 GCN ACCELERATOR
echo ==================================================================

set VIVADO_CMD=
if exist "C:\Xilinx\Vivado\2024.2\bin\vivado.bat" set VIVADO_CMD="C:\Xilinx\Vivado\2024.2\bin\vivado.bat"
if exist "D:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat" set VIVADO_CMD="D:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
if "%VIVADO_CMD%"=="" set VIVADO_CMD=vivado

echo Found Vivado at: %VIVADO_CMD%
cd /d "%~dp0"
%VIVADO_CMD% -mode batch -source run_synth.tcl

echo ==================================================================
echo   Reports generated in: %~dp0output\
echo ==================================================================
pause
