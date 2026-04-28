@echo off
chcp 65001 > nul

:menu
cls
echo ==========================================================
echo  请选择您需要加载的 Quanser Interactive Labs 场景：
echo ==========================================================
echo.
echo  1. Cityscape
echo  2. Cityscape Lite
echo  3. Townscape
echo  4. Townscape Lite
echo  5. Open Road
echo  6. Plane
echo  7. Studio
echo  8. Warehouse
echo.
echo  0. 退出
echo.
echo ==========================================================

set /p "choice=请输入您的选项 (0-8) 并按回车键: "

if /i "%choice%"=="1" set "scene=Cityscape"
if /i "%choice%"=="2" set "scene=CityscapeLite"
if /i "%choice%"=="3" set "scene=Townscape"
if /i "%choice%"=="4" set "scene=TownscapeLite"
if /i "%choice%"=="5" set "scene=OpenRoad"
if /i "%choice%"=="6" set "scene=Plane"
if /i "%choice%"=="7" set "scene=Studio"
if /i "%choice%"=="8" set "scene=Warehouse"
if /i "%choice%"=="0" goto :eof

if defined scene (
    echo.
    echo 正在唤起场景: %scene%...
    start "" "%QUARC_DIR%\..\Quanser Interactive Labs\Quanser Interactive Labs.exe" -loadmodule %scene%
    goto :eof
) else (
    echo.
    echo 无效输入，请按任意键后重新选择。
    pause > nul
    goto menu
)