@echo off
setlocal enabledelayedexpansion

echo ==================================================================
echo   OPENING VIVADO GUI WITH SCHEMATIC AND PROJECT
echo ==================================================================

set VIVADO_CMD=
if exist "C:\Xilinx\Vivado\2024.2\bin\vivado.bat" set VIVADO_CMD="C:\Xilinx\Vivado\2024.2\bin\vivado.bat"
if exist "D:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat" set VIVADO_CMD="D:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
if "%VIVADO_CMD%"=="" set VIVADO_CMD=vivado

echo Found Vivado at: %VIVADO_CMD%
cd /d "%~dp0"
%VIVADO_CMD% -mode gui -source run_synth.tcl

pause
