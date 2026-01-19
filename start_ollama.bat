@echo off
echo Starting Ollama server and pulling DeepSeek model...
echo.

:: Start Ollama server in background
start cmd /k "ollama serve"

:: Wait a bit for server to start
timeout /t 5 /nobreak

:: Pull the DeepSeek model
ollama pull deepseek-r1:1.5b

echo.
echo Ollama server is running and DeepSeek model is ready!
echo You can now run Syndgen with: python main.py --mode export --batch-size 20
echo.

:: Keep the window open
pause
