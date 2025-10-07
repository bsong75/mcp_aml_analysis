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

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    """Chat with Gemma3 via MCP or direct Ollama"""
    normalized_message = " ".join(request.message.lower().split())
    
    # Handle tools command
    if any(keyword in normalized_message for keyword in ["tool", "tools", "help", "/help"]):
        tools_response = await list_tools()
        formatted_response = "🛠️ **Available Tools:**\n\n"
        for tool in tools_response["tools"]:
            formatted_response += f"• **{tool['name']}**: {tool['description']}\n"
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

        system_message = """You are ATLAS (Agentic Toolkit for Learning and Advanced Solutions), an advanced AI agent. You are sophisticated, witty, efficient, and always ready to help. Speak with confidence and a touch of dry humor when appropriate, but remain professional and helpful. You have several MCP tools to offer including:
- Exploratory Data Analysis (EDA)
- Feature Analysis
- Fan-in Analysis for transaction graphs

When users ask for help or tools, guide them to use the appropriate commands."""

        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": request.message}
            ],
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=300)
        
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

def direct_exploratory_data_analysis(csv_filename=None):
    """Direct EDA without MCP"""
    import pandas as pd
    import numpy as np
    import glob
    import os
    
    try:
        # Define the features folder
        features_folder = "/app/graph_features_files"

        # If no specific file provided, find most recent CSV file
        if not csv_filename:
            csv_files = glob.glob(f"{features_folder}/*.csv")
            if not csv_files:
                return "❌ No CSV files found in graph_features_files folder"
            # Sort by modification time, most recent first
            csv_files.sort(key=os.path.getmtime, reverse=True)
            csv_file = csv_files[0]  # Use most recently modified CSV
            csv_filename = os.path.basename(csv_file)
            print(f"📊 EDA using most recent CSV: {csv_filename}")
        else:
            csv_file = os.path.join(features_folder, csv_filename)
            if not os.path.exists(csv_file):
                return f"❌ CSV file '{csv_filename}' not found in graph_features_files folder"

        # Load the CSV
        df = pd.read_csv(csv_file)
        
        # Basic dataset info
        rows, cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df) * 100).round(2)
        
        # Data types summary
        dtype_summary = df.dtypes.value_counts()
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # Numeric data insights
        numeric_insights = {}
        if numeric_cols:
            numeric_df = df[numeric_cols]
            
            # Statistical summary
            stats = numeric_df.describe()
            
            # Skewness and kurtosis
            skewness = numeric_df.skew().round(3)
            kurtosis = numeric_df.kurtosis().round(3)
            
            # Correlation analysis
            correlation = numeric_df.corr()
            high_corr_pairs = []
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    corr_val = correlation.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'var1': correlation.columns[i],
                            'var2': correlation.columns[j],
                            'correlation': round(corr_val, 3)
                        })
            
            # Outlier detection (using IQR method)
            outliers_summary = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    outliers_summary[col] = {
                        'count': len(outliers),
                        'percentage': round(len(outliers) / len(df) * 100, 2)
                    }
        
        # Categorical data insights
        categorical_insights = {}
        if categorical_cols:
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                categorical_insights[col] = {
                    'unique_values': unique_vals,
                    'most_frequent': most_frequent,
                    'cardinality': 'High' if unique_vals > len(df) * 0.5 else 'Low'
                }
        
        # Data quality assessment
        duplicate_rows = df.duplicated().sum()
        complete_cases = df.dropna().shape[0]
        data_completeness = round(complete_cases / rows * 100, 2)
        
        # Generate comprehensive EDA report
        eda_report = f"""🔍 **Comprehensive Exploratory Data Analysis**

**📋 Dataset Overview:**
• File: {csv_filename}
• Dimensions: {rows:,} rows × {cols} columns
• Memory Usage: {memory_usage:.2f} MB
• Data Completeness: {data_completeness}%
• Duplicate Rows: {duplicate_rows:,}

**📊 Column Types:**
• Numeric: {len(numeric_cols)} columns
• Categorical: {len(categorical_cols)} columns
• Data Types: {dict(dtype_summary)}

**🔢 Numeric Data Insights:**"""

        if numeric_cols:
            eda_report += f"""
• Variables: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
• Highly Skewed Features: {', '.join([col for col in numeric_cols if abs(skewness.get(col, 0)) > 1][:3]) or 'None'}
• High Correlations: {len(high_corr_pairs)} pairs found"""
            
            if high_corr_pairs:
                eda_report += f"""
• Top Correlations:"""
                for pair in high_corr_pairs[:3]:
                    eda_report += f"""
  - {pair['var1']} ↔ {pair['var2']}: {pair['correlation']}"""
            
            if outliers_summary:
                eda_report += f"""
• Outliers Detected:"""
                for col, info in list(outliers_summary.items())[:3]:
                    eda_report += f"""
  - {col}: {info['count']} outliers ({info['percentage']}%)"""
        
        if categorical_cols:
            eda_report += f"""

**📝 Categorical Data Insights:**"""
            for col, info in list(categorical_insights.items())[:3]:
                eda_report += f"""
• {col}: {info['unique_values']} unique values, most frequent: '{info['most_frequent']}'"""

        eda_report += f"""

**⚠️ Data Quality Issues:**
• Missing Values: {missing_data.sum():,} total"""
        
        missing_cols = missing_data[missing_data > 0]
        if len(missing_cols) > 0:
            eda_report += f"""
• Columns with Missing Data:"""
            for col in missing_cols.head(3).index:
                eda_report += f"""
  - {col}: {missing_data[col]:,} ({missing_percent[col]:.1f}%)"""

        eda_report += f"""

**🔗 Interactive Analysis:**
• [Open EDA Dashboard](http://localhost:8001/dashboard?file={csv_filename}) - Interactive crossfilter analysis"""

        return eda_report
        
    except Exception as e:
        return f"❌ Error in EDA: {str(e)}"

def direct_csv_feature_analysis(csv_filename=None):
    """Direct CSV feature analysis without MCP"""
    import pandas as pd
    import glob
    import os
    
    try:
        # Define the features folder
        features_folder = "/app/graph_features_files"

        # If no specific file provided, find most recent CSV file
        if not csv_filename:
            csv_files = glob.glob(f"{features_folder}/*.csv")
            if not csv_files:
                return "❌ No CSV files found in graph_features_files folder"
            # Sort by modification time, most recent first
            csv_files.sort(key=os.path.getmtime, reverse=True)
            csv_file = csv_files[0]  # Use most recently modified CSV
            csv_filename = os.path.basename(csv_file)
            print(f"📊 Feature Analysis using most recent CSV: {csv_filename}")
        else:
            csv_file = os.path.join(features_folder, csv_filename)
            if not os.path.exists(csv_file):
                return f"❌ CSV file '{csv_filename}' not found in graph_features_files folder"

        # Load and analyze the CSV
        df = pd.read_csv(csv_file)
        
        # Get basic info
        rows, cols = df.shape
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if 'Id' not in col and 'nodeId' not in col]
        
        # Calculate summary statistics
        if feature_cols:
            stats = df[feature_cols].describe()
            mean_vals = stats.loc['mean'].round(3)
            std_vals = stats.loc['std'].round(3)
            
            # Find most variable features (highest coefficient of variation)
            cv = (std_vals / mean_vals).sort_values(ascending=False)
            top_variable = cv.head(3).index.tolist()
        else:
            top_variable = []
        
        # Create summary
        summary = f"""✅ **CSV Feature Analysis Complete**

**File:** {csv_filename}
**Dataset Size:** {rows} rows × {cols} columns
**Features Analyzed:** {len(feature_cols)} numeric features

**Key Insights:**
• Total features: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}
• Most variable features: {', '.join(top_variable) if top_variable else 'None'}
• Data completeness: {100 - (df.isnull().sum().sum() / (rows * cols) * 100):.1f}%

🎯 **Interactive Analysis Available:**
• [Open Gradio Feature Analyzer](http://localhost:7860) - Detailed feature analysis with visualizations

The Gradio analyzer provides:
• 📈 Statistical summaries and quality assessments
• 📦 Distribution plots, boxplots, and correlation heatmaps"""
        
        return summary
        
    except Exception as e:
        return f"❌ Error analyzing CSV: {str(e)}"

@app.post("/api/csv-feature-analysis")
async def csv_feature_analysis_endpoint(request: dict = {}):
    """Analyze CSV features and provide summary with Gradio link"""
    print("📊 Frontend: CSV feature analysis endpoint called!")
    print(f"Request data: {request}")

    csv_filename = request.get('csv_filename') if request else None

    # Try MCP first, fall back to direct function call
    try:
        # Only include csv_filename in arguments if it's provided
        args = {"csv_filename": csv_filename} if csv_filename else {}
        mcp_result = await call_mcp_tool("csv_feature_analysis", args)
        
        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            raise Exception(mcp_result.get("error", "MCP tool failed"))
    except Exception as e:
        print(f"MCP error: {e}, falling back to direct function...")
        
        # Fallback to direct function call
        try:
            result = direct_csv_feature_analysis(csv_filename)
            return {"response": result, "status": "success"}
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
        
        # Immediately analyze the uploaded file
        result = direct_csv_feature_analysis(file.filename)
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File '{file.filename}' uploaded successfully",
            "analysis": result
        }
        
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"❌ Upload error: {error_msg}")
        return {"error": error_msg, "status": "upload_error"}

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

    # Try MCP first, fall back to direct function call
    try:
        # Only include csv_filename in arguments if it's provided
        args = {"csv_filename": csv_filename} if csv_filename else {}
        mcp_result = await call_mcp_tool("exploratory_data_analysis", args)
        
        if mcp_result["success"]:
            return {"response": mcp_result["result"], "status": "success"}
        else:
            raise Exception(mcp_result.get("error", "MCP tool failed"))
    except Exception as e:
        print(f"MCP error: {e}, falling back to direct function...")
        
        # Fallback to direct function call
        try:
            result = direct_exploratory_data_analysis(csv_filename)
            return {"response": result, "status": "success"}
        except Exception as e:
            error_msg = f"EDA failed: {str(e)}"
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
            {"name": "fan_in_analysis", "description": "Analyze transaction graph for fan-in patterns (money laundering detection)"},
            {"name": "neo4j_visualizations", "description": "Show the GraphDB Details"}
        ]
    }

@app.get("/dashboard")
async def eda_dashboard(request: Request, file: str = None):
    """Serve the interactive EDA dashboard"""
    return templates.TemplateResponse("eda_dashboard.html", {
        "request": request,
        "filename": file
    })

@app.get("/api/csv-data")
async def get_csv_data(filename: str = None):
    """Get CSV data as JSON for the dashboard"""
    print(f"📊 CSV data request for: {filename}")
    
    try:
        # Define the features folder
        features_folder = "/app/graph_features_files"
        
        # If no specific file provided, find most recent CSV file
        if not filename:
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
    # print("Starting frontend server on http://localhost:8001")
    # print(f"Neo4j URI: {os.getenv('NEO4J_URI', 'Not set')}")
    # print(f"Ollama Host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    # print(f"MCP Server: {MCP_SERVER_URL}")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        print("\n MCP Server Shutting down...")
        stop_mcp_server()
    except Exception as e:
        print(f" ======== Frontend error: {e}")
        stop_mcp_server()
        raise