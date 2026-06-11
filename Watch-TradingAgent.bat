@echo off
chcp 65001 >nul
title TradingAgent Monitor
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\watch.ps1"
