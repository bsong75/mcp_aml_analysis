FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "frontend_app.py"]
#CMD ["uvicorn", "frontend_app:app", "--host", "0.0.0.0", "--port", "8000"]
# Start both MCP server and frontend app
# Use exec to ensure proper signal handling and process cleanup
#CMD ["sh", "-c", "echo 'Starting MCP server...' && python -u mcp_server.py & MCP_PID=$! && sleep 5 && echo 'Starting frontend app...' && exec python -u frontend_app.py"]
