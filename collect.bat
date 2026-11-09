@echo off
cd /d G:\NoSock\stock\penny
start "" /b /low /wait python collect.py >> collect.log 2>&1
