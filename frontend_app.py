#!/usr/bin/env python3
"""
Frontend FastAPI app with web interface and MCP integration
"""
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
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

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3"
    session_id: str = "default"  # Add session ID to track conversations

# Chat history storage (in-memory, per session)
# Structure: {session_id: [{"role": "user/assistant", "content": "..."}]}
chat_sessions = {}
MAX_HISTORY_LENGTH = 10  # Keep last 10 exchanges (20 messages)

def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]

def add_to_chat_history(session_id: str, role: str, content: str):
    """Add a message to chat history"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({"role": role, "content": content})

    # Keep only last N exchanges (2N messages)
    if len(chat_sessions[session_id]) > MAX_HISTORY_LENGTH * 2:
        chat_sessions[session_id] = chat_sessions[session_id][-(MAX_HISTORY_LENGTH * 2):]

def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in chat_sessions:
        chat_sessions[session_id] = []

# MCP HTTP client configuration
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://mcp-app:8000')

async def call_mcp_tool(tool_name: str, arguments: dict = None):
    """Call an MCP tool via HTTP POST to FastMCP streamable endpoint"""
    if arguments is None:
        arguments = {}

    try:
        # FastMCP streamable HTTP protocol with session management
        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }

            # Step 1: Initialize the session
            init_request = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "frontend", "version": "1.0.0"}
                }
            }

            init_response = await client.post(
                f"{MCP_SERVER_URL}/mcp",
                json=init_request,
                headers=headers
            )

            if init_response.status_code != 200:
                return {"success": False, "error": f"Init failed: {init_response.status_code} {init_response.text[:200]}"}

            # Extract session ID from response headers (FastMCP uses 'mcp-session-id')
            session_id = init_response.headers.get('mcp-session-id')

            if not session_id:
                return {"success": False, "error": "No session ID received from MCP server"}

            print(f"Got MCP session ID: {session_id}")

            # Add session ID to headers for subsequent requests
            headers['mcp-session-id'] = session_id

            # Step 1.5: Send initialized notification (required by MCP protocol)
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }

            await client.post(
                f"{MCP_SERVER_URL}/mcp",
                json=initialized_notification,
                headers=headers
            )

            # Step 2: Call the tool
            tool_request = {
                "jsonrpc": "2.0",
                "id": "tool-call-1",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            tool_response = await client.post(
                f"{MCP_SERVER_URL}/mcp",
                json=tool_request,
                headers=headers
            )

            if tool_response.status_code == 200:
                # Parse SSE response if content-type is text/event-stream
                response_text = tool_response.text

                # Parse SSE format: "event: message\ndata: {json}\n\n"
                if 'event: message' in response_text:
                    # Extract JSON from SSE data field
                    lines = response_text.split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            json_str = line[6:]  # Remove 'data: ' prefix
                            import json
                            result = json.loads(json_str)
                            break
                    else:
                        result = None
                else:
                    # Try parsing as regular JSON
                    try:
                        result = tool_response.json()
                    except:
                        return {"success": False, "error": f"Cannot parse response: {response_text[:200]}"}

                if not result:
                    return {"success": False, "error": f"No data in SSE response: {response_text[:200]}"}

                # Check for JSON-RPC error
                if "error" in result:
                    return {"success": False, "error": f"Tool error: {result['error'].get('message', str(result['error']))}"}

                # Extract result
                if "result" in result:
                    result_data = result["result"]
                    # Extract text content
                    if isinstance(result_data, dict) and "content" in result_data:
                        content = result_data["content"]
                        if isinstance(content, list) and len(content) > 0:
                            text = content[0].get("text", str(content[0]))
                            return {"success": True, "result": text}
                    return {"success": True, "result": str(result_data)}

                return {"success": False, "error": f"Unexpected response format: {str(result)[:200]}"}
            else:
                return {"success": False, "error": f"HTTP {tool_response.status_code}: {tool_response.text[:200]}"}

    except httpx.ConnectError:
        return {"success": False, "error": f"Cannot connect to MCP server at {MCP_SERVER_URL}"}
    except Exception as e:
        import traceback
        error_detail = f"MCP client error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        return {"success": False, "error": f"MCP client error: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with Gemma3 via MCP or direct Ollama with conversation history"""
    normalized_message = " ".join(request.message.lower().split())

    # Handle tools command
    if any(keyword in normalized_message for keyword in ["tool", "tools", "help", "/help"]):
        tools_response = await list_tools()
        formatted_response = "🛠️ **Available Tools:**\n\n"
        for tool in tools_response["tools"]:
            formatted_response += f"• **{tool['name']}**: {tool['description']}\n"
        return {"response": formatted_response}

    # Get chat history for this session
    history = get_chat_history(request.session_id)

    # Fallback to direct Ollama connection (with history support)
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        url = f"{ollama_host}/v1/chat/completions"

        system_message = """You are ATLAS (Agentic Toolkit for Learning and Advanced Solutions), an advanced AI agent specialized in CBP Agriculture. You are sophisticated, witty, efficient, and always ready to help. Speak with confidence and a touch of dry humor when appropriate, but remain professional and helpful. You have several MCP tools to offer including:
- Exploratory Data Analysis (EDA)
- CSV Feature Analysis
- Neo4j Graph Visualization
- CBP Agriculture Acronym Lookup

When users ask for help or tools, guide them to use the appropriate commands.

IMPORTANT: You have access to the conversation history. When users refer to previous topics (like "the state I asked about", "that port", "those countries"), use the context from earlier messages to provide accurate responses."""

        # Build messages array with history
        messages = [{"role": "system", "content": system_message}]

        # Add conversation history
        messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": False
        }

        response = requests.post(url, json=payload, timeout=300)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                assistant_response = result['choices'][0]['message']['content']

                # Save to history
                add_to_chat_history(request.session_id, "user", request.message)
                add_to_chat_history(request.session_id, "assistant", assistant_response)

                return {"response": assistant_response}
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

@app.post("/api/neo4j-visualization")
async def neo4j_visualization_endpoint():
    print("🚀 Frontend: Neo4j visualization endpoint called!")

    try:
        mcp_result = await call_mcp_tool("neo4j_visualization", {})
        print(f"MCP Result: {mcp_result}")

        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            error_detail = mcp_result.get("error", "Unknown error")
            print(f"MCP Error: {error_detail}")
            return {"error": f"Failed to get Neo4j visualization: {error_detail}", "status": "error"}
    except Exception as e:
        error_msg = f"Neo4j visualization failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}


@app.post("/api/csv-feature-analysis")
async def csv_feature_analysis_endpoint(request: dict = {}):
    """Analyze CSV features and provide summary with Gradio link"""
    print("📊 Frontend: CSV feature analysis endpoint called!")
    print(f"Request data: {request}")

    csv_filename = request.get('csv_filename') if request else None

    try:
        # Only include csv_filename in arguments if it's provided
        args = {"csv_filename": csv_filename} if csv_filename else {}
        mcp_result = await call_mcp_tool("csv_feature_analysis", args)

        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            error_detail = mcp_result.get("error", "Unknown error")
            print(f"MCP Error: {error_detail}")
            return {"error": f"CSV analysis failed: {error_detail}", "status": "error"}
    except Exception as e:
        error_msg = f"CSV analysis failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}

@app.post("/api/upload-csv")
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload CSV file for analysis"""
    print(f"📂 File upload: {file.filename}")

    # Validate file type
    if not file.filename.endswith('.csv'):
        return {"error": "Only CSV files are allowed", "status": "invalid_file"}

    try:
        # Ensure upload directory exists
        upload_dir = "/app/graph_features_files"
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        print(f"✅ File saved: {file_path}")

        # Immediately analyze the uploaded file via MCP
        mcp_result = await call_mcp_tool("csv_feature_analysis", {"csv_filename": file.filename})

        analysis_result = mcp_result.get("result", "Analysis pending") if mcp_result.get("success") else f"Analysis error: {mcp_result.get('error', 'Unknown error')}"

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File '{file.filename}' uploaded successfully",
            "analysis": analysis_result
        }

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"❌ Upload error: {error_msg}")
        return {"error": error_msg, "status": "upload_error"}

@app.post("/api/clear-chat")
async def clear_chat(request: dict = {}):
    """Clear chat history for a session"""
    session_id = request.get('session_id', 'default')
    clear_chat_history(session_id)
    return {"status": "success", "message": f"Chat history cleared for session {session_id}"}

@app.post("/api/delete-csv")
async def delete_csv_file(request: dict = {}):
    """Delete uploaded CSV file"""
    print(f"🗑️ File delete request: {request}")

    filename = request.get('filename')
    if not filename:
        return {"error": "No filename provided", "status": "invalid_request"}
    
    try:
        # Define file path
        file_path = os.path.join("/app/graph_features_files", filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File '{filename}' not found", "status": "file_not_found"}
        
        # Delete the file
        os.remove(file_path)
        print(f"✅ File deleted: {file_path}")
        
        return {
            "status": "success",
            "filename": filename,
            "message": f"File '{filename}' deleted successfully"
        }
        
    except Exception as e:
        error_msg = f"Delete failed: {str(e)}"
        print(f"❌ Delete error: {error_msg}")
        return {"error": error_msg, "status": "delete_error"}

@app.post("/api/exploratory-data-analysis")
async def eda_endpoint(request: dict = {}):
    """Perform comprehensive Exploratory Data Analysis"""
    print("🔍 Frontend: EDA endpoint called!")
    print(f"Request data: {request}")

    csv_filename = request.get('csv_filename') if request else None

    try:
        # Only include csv_filename in arguments if it's provided
        args = {"csv_filename": csv_filename} if csv_filename else {}
        mcp_result = await call_mcp_tool("exploratory_data_analysis", args)

        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            error_detail = mcp_result.get("error", "Unknown error")
            print(f"MCP Error: {error_detail}")
            return {"error": f"EDA failed: {error_detail}", "status": "error"}
    except Exception as e:
        error_msg = f"EDA failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}

@app.post("/api/acronym-lookup")
async def acronym_lookup_endpoint(request: dict = {}):
    """Look up CBP Agriculture acronym definition"""
    print("🔤 Frontend: Acronym lookup endpoint called!")
    print(f"Request data: {request}")

    acronym = request.get('acronym', '').strip()

    if not acronym:
        return {"error": "No acronym provided", "status": "invalid_request"}

    try:
        # Call MCP acronym lookup tool
        mcp_result = await call_mcp_tool("acronym_lookup", {"acronym": acronym})

        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            return {"error": mcp_result.get("error", "Unknown error"), "status": "error"}

    except Exception as e:
        error_msg = f"Acronym lookup failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}

@app.post("/api/acronym-update")
async def acronym_update_endpoint(request: dict = {}):
    """Add or update a CBP Agriculture acronym"""
    print("➕ Frontend: Acronym update endpoint called!")
    print(f"Request data: {request}")

    acronym = request.get('acronym', '').strip()
    definition = request.get('definition', '').strip()

    if not acronym or not definition:
        return {"error": "Both acronym and definition are required", "status": "invalid_request"}

    try:
        # Call MCP acronym update tool
        mcp_result = await call_mcp_tool("acronym_update", {"acronym": acronym, "definition": definition})

        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            return {"error": mcp_result.get("error", "Unknown error"), "status": "error"}

    except Exception as e:
        error_msg = f"Acronym update failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}

@app.post("/api/acronym-delete")
async def acronym_delete_endpoint(request: dict = {}):
    """Delete a CBP Agriculture acronym"""
    print("🗑️ Frontend: Acronym delete endpoint called!")
    print(f"Request data: {request}")

    acronym = request.get('acronym', '').strip()

    if not acronym:
        return {"error": "No acronym provided", "status": "invalid_request"}

    try:
        # Call MCP acronym delete tool
        mcp_result = await call_mcp_tool("acronym_delete", {"acronym": acronym})

        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            return {"error": mcp_result.get("error", "Unknown error"), "status": "error"}

    except Exception as e:
        error_msg = f"Acronym delete failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}

@app.post("/api/globe-visualization")
async def globe_visualization_endpoint():
    """Generate 3D Globe choropleth visualization via MCP"""
    print("🌍 Frontend: Globe visualization endpoint called!")

    try:
        # Call MCP tool
        mcp_result = await call_mcp_tool("globe_visualization", {})
        print(f"MCP Result: {mcp_result}")

        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            error_detail = mcp_result.get("error", "Unknown error")
            print(f"MCP Error: {error_detail}")
            return {"error": f"Failed to generate globe visualization: {error_detail}", "status": "error"}
    except Exception as e:
        error_msg = f"Globe visualization failed: {str(e)}"
        print(f"Frontend Error: {error_msg}")
        return {"error": error_msg, "status": "execution_error"}

@app.get("/api/tools")
async def list_tools():
    """List available tools"""
    # Since FastMCP doesn't support custom endpoints, return hardcoded tools
    return {
        "tools": [
            {"name": "csv_feature_analysis", "description": "Analyze CSV features with interactive Gradio visualizations"},
            {"name": "exploratory_data_analysis", "description": "Comprehensive EDA with correlations, outliers, and data quality insights"},
            {"name": "chat_gemma3", "description": "Chat with Gemma3 via Ollama"},
            {"name": "neo4j_visualizations", "description": "Show the GraphDB Details"},
            {"name": "acronym_lookup", "description": "Look up CBP Agriculture acronyms with fuzzy matching"},
            {"name": "acronym_update", "description": "Add or update CBP Agriculture acronyms in the database"},
            {"name": "acronym_delete", "description": "Delete CBP Agriculture acronyms from the database"},
            {"name": "globe_visualization", "description": "3D Globe choropleth visualization by country code"}
        ]
    }

@app.get("/dashboard")
async def eda_dashboard(request: Request, file: str = None):
    """Serve the interactive EDA dashboard"""
    return templates.TemplateResponse("eda_dashboard.html", {
        "request": request,
        "filename": file
    })

@app.get("/globe")
async def globe_visualization_page(request: Request):
    """Serve the 3D Globe choropleth visualization page"""
    return templates.TemplateResponse("globe_visualization.html", {
        "request": request
    })

@app.get("/cbp_acronyms.json")
async def get_cbp_acronyms():
    """Serve the CBP acronyms JSON file"""
    import json
    acronym_file = os.path.join(os.path.dirname(__file__), 'cbp_acronyms.json')
    try:
        with open(acronym_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "CBP acronyms file not found"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/csv-data")
async def get_csv_data(filename: str = None):
    """Get CSV data as JSON for the dashboard"""
    print(f"📊 CSV data request for: {filename}")

    try:
        # Define the features folder
        features_folder = "/app/graph_features_files"

        # If no specific file provided (None or empty string), find most recent CSV file
        if not filename or filename.strip() == "":
            import glob
            csv_files = glob.glob(f"{features_folder}/*.csv")
            if not csv_files:
                return {"error": "No CSV files found in graph_features_files folder"}
            # Sort by modification time, most recent first
            csv_files.sort(key=os.path.getmtime, reverse=True)
            csv_file = csv_files[0]  # Use most recently modified CSV
            filename = os.path.basename(csv_file)
            print(f"📊 Using most recent CSV: {filename}")
        else:
            csv_file = os.path.join(features_folder, filename)
            if not os.path.exists(csv_file):
                return {"error": f"CSV file '{filename}' not found in graph_features_files folder"}
        
        # Load CSV data
        import pandas as pd
        df = pd.read_csv(csv_file)

        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Convert to JSON-serializable format
        data = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                # Handle NaN values and ensure JSON serializable
                if pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, (int, float, str, bool)):
                    row_dict[col] = value
                else:
                    row_dict[col] = str(value)
            data.append(row_dict)
        
        return {
            "status": "success",
            "filename": filename,
            "data": data,
            "rows": len(data),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        error_msg = f"Error loading CSV data: {str(e)}"
        print(f"❌ CSV data error: {error_msg}")
        return {"error": error_msg}



if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        print("\n MCP Server Shutting down...")
        stop_mcp_server()
    except Exception as e:
        print(f" ======== Frontend error: {e}")
        stop_mcp_server()
        raise