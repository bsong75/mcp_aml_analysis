"""
FastAPI app converted to MCP Server with Fan-In Analysis 
Now using HTTP transport for Docker compatibility
"""

import math
import requests
import os
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from fanin_standalone import standalone_fan_in_analysis

@tool
def fan_in_analysis(neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None, output_file: str = None) -> str:
    """Perform fan-in analysis
    Args:
        neo4j info
        output_file: Optional custom output file path for results CSV
    Returns:
        String with analysis summary and file location or error message
    """
    print("🚀 FAN-IN ANALYSIS TOOL CALLED!")
    
    # Call the standalone fan-in analysis function
    result = standalone_fan_in_analysis(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        output_file=output_file
    )
    
    print("🎉 MCP Fan-in analysis completed successfully!")
    return result

@tool
def chat_gemma3(message: str, model: str = "gemma3", timeout: int = 3000) -> str:
    """Chat with Gemma3 via Ollama
    Args:
        message: The message to send to the AI model
        model: The model to use (default: gemma3)
        timeout: Request timeout in seconds (default: 3000)
    Returns:
        String response from the AI model or error message
    """
    print("💬 CHAT_GEMMA3 TOOL CALLED!")
    print(f"Message: {message}")
    print(f"Model: {model}")
    
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        url = f"{ollama_host}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False
        }
        
        print(f"Attempting to connect to Ollama at: {url}")
        print(f"Using timeout: {timeout} seconds")
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        elif response.status_code == 404:
            return f"Error: Model '{model}' not found. Please make sure Gemma3 is installed in Ollama."
        else:
            return f"Error: Ollama returned status {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return f"Error: Cannot connect to Ollama at {ollama_host}. Make sure Ollama is running."
    except requests.exceptions.timeout:
        return f"Error: Request timed out after {timeout} seconds. The analysis might be processing a very large dataset."
    except KeyError:
        return "Error: Unexpected response format from Ollama"
    except Exception as e:
        return f"Error: {str(e)}"

# Convert to MCP tools
fanin_tool = to_fastmcp(fan_in_analysis)
chat_tool = to_fastmcp(chat_gemma3)

# Create MCP server
mcp = FastMCP(name="Money Laundering Analysis MCP Server", 
              tools=[fanin_tool, chat_tool] 
            )

if __name__ == "__main__":
    print("🚀 Starting Money Laundering Analysis MCP Server with HTTP transport")
    print(f"Neo4j URI: {os.getenv('NEO4J_URI', 'Not set')}")
    print(f"Neo4j User: {os.getenv('NEO4J_USERNAME', 'Not set')}")
    print(f"Ollama host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print("\nAvailable tools:")
    print("- fan_in_analysis: Perform fan-in analysis on Neo4j graph")
    print("- chat_gemma3: Chat with Gemma3 via Ollama")
    print("🔧 Debug mode enabled - will show which tools are called")
    print("🌐 MCP Server will be available on http://0.0.0.0:8000")
    
    # Use the correct FastMCP run method
    mcp.run()