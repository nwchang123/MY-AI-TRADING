@echo off
chcp 65001 >nul
title TradingAgent 一键启动
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_all.ps1"
