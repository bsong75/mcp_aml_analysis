#!/usr/bin/env python3
"""
Frontend FastAPI app with web interface and MCP integration
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import asyncio
import sys
import os
import subprocess
import time
import threading
from pathlib import Path
import uvicorn
import httpx
from contextlib import asynccontextmanager
from fanin_standalone import standalone_fan_in_analysis

# Global variable to hold the MCP server process
mcp_process = None

def start_mcp_server():
    """Start the MCP server as a subprocess"""
    global mcp_process
    
    print("🚀 Starting MCP server subprocess...")
    try:
        mcp_process = subprocess.Popen(
            [sys.executable, "mcp_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ MCP server started with PID: {mcp_process.pid}")
        
        # Start a thread to monitor MCP server output
        def monitor_mcp_output():
            if mcp_process and mcp_process.stdout:
                for line in iter(mcp_process.stdout.readline, ''):
                    if line:
                        print(f"[MCP] {line.strip()}")
        
        monitor_thread = threading.Thread(target=monitor_mcp_output, daemon=True)
        monitor_thread.start()
        
        # Wait for the server to start
        time.sleep(3)
        return True
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        return False

def stop_mcp_server():
    """Stop the MCP server subprocess"""
    global mcp_process
    if mcp_process:
        print("🛑 Stopping MCP server...")
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=5)
            print("✅ MCP server stopped")
        except subprocess.TimeoutExpired:
            print("⚠️ MCP server didn't stop gracefully, killing...")
            mcp_process.kill()
        mcp_process = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    print("🔧 Frontend starting up...")
    start_mcp_server()
    
    yield
    
    # Shutdown
    print("🔧 Frontend shutting down...")
    stop_mcp_server()

app = FastAPI(title="MCP Tools Frontend", lifespan=lifespan)

templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3"

class FanInRequest(BaseModel):
    neo4j_uri: str = None
    neo4j_user: str = None
    neo4j_password: str = None
    output_file: str = None

# MCP HTTP client configuration
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://127.0.0.1:8000')

async def call_mcp_tool(tool_name: str, arguments: dict = None):
    """Call an MCP tool via HTTP"""
    if arguments is None:
        arguments = {}
        
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{MCP_SERVER_URL}/tools/{tool_name}/call",
                json={"arguments": arguments}
            )
            
            if response.status_code == 200:
                result = response.json()
                if "content" in result and len(result["content"]) > 0:
                    return {"success": True, "result": result["content"][0].get("text", str(result))}
                else:
                    return {"success": True, "result": str(result)}
            else:
                return {"success": False, "error": f"MCP server error {response.status_code}: {response.text}"}
                
    except httpx.ConnectError:
        return {"success": False, "error": f"Cannot connect to MCP server at {MCP_SERVER_URL}"}
    except Exception as e:
        return {"success": False, "error": f"MCP client error: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with Gemma3 via MCP or direct Ollama"""
    normalized_message = " ".join(request.message.lower().split())
    
    # Handle tools command
    if any(keyword in normalized_message for keyword in ["tool", "tools", "help", "/help"]):
        tools_response = await list_tools()
        formatted_response = "🛠️ **Available Tools:**\n\n"
        for tool in tools_response["tools"]:
            formatted_response += f"• **{tool['name']}**: {tool['description']}\n\n"
        return {"response": formatted_response}
    
    # Try MCP first, fall back to direct Ollama
    try:
        mcp_result = await call_mcp_tool("chat_gemma3", {
            "message": request.message,
            "model": request.model
        })
        
        if mcp_result["success"]:
            return {"response": mcp_result["result"]}
    except Exception as e:
        print(f"MCP error: {e}, falling back to direct Ollama...")
    
    # Fallback to direct Ollama connection
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        url = f"{ollama_host}/v1/chat/completions"
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.message}],
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return {"response": result['choices'][0]['message']['content']}
            else:
                return {"error": f"Unexpected response format: {str(result)[:200]}..."}
        elif response.status_code == 404:
            return {"error": f"Model '{request.model}' not found. Try: ollama pull {request.model}"}
        else:
            return {"error": f"Ollama error {response.status_code}: {response.text[:200]}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to Ollama at {ollama_host}. Make sure Ollama is running and accessible."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@app.post("/api/fan-in-analysis")
async def fan_in_analysis_endpoint(request: FanInRequest = None):
    print("🚀 Frontend: Fan-in analysis endpoint called!")
    
    # Try MCP first, fall back to direct function call
    try:
        mcp_result = await call_mcp_tool("fan_in_analysis", {
            "neo4j_uri": request.neo4j_uri if request else None,
            "neo4j_user": request.neo4j_user if request else None,
            "neo4j_password": request.neo4j_password if request else None,
            "output_file": request.output_file if request else None
        })
        
        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
    except Exception as e:
        print(f"MCP error: {e}, falling back to direct function...")
    
    # Fallback to direct function call
    try:
        result = standalone_fan_in_analysis()
        return {"response": result, "status": "success"}
    except Exception as e:
        error_msg = f"Fan-in analysis failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}

@app.get("/api/tools")
async def list_tools():
    """List available tools"""
    # Since FastMCP doesn't support custom endpoints, return hardcoded tools
    return {
        "tools": [
            {"name": "chat_gemma3", "description": "Chat with Gemma3 via Ollama"},
            {"name": "fan_in_analysis", "description": "Analyze transaction graph for fan-in patterns (money laundering detection)"}
        ]
    }



if __name__ == "__main__":
    print("Starting frontend server on http://localhost:8001")
    print(f"Neo4j URI: {os.getenv('NEO4J_URI', 'Not set')}")
    print(f"Ollama Host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"MCP Server: {MCP_SERVER_URL}")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        print("\n MCP Server Shutting down...")
        stop_mcp_server()
    except Exception as e:
        print(f" ======== Frontend error: {e}")
        stop_mcp_server()
        raise