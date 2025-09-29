#!/usr/bin/env python3
"""
Simple MCP Client with Fan-In Analysis (HTTP version)
"""
import asyncio
import httpx

async def call_mcp_tool(tool_name: str, arguments: dict = None, base_url="http://localhost:8000"):
    """Call an MCP tool via HTTP"""
    if arguments is None:
        arguments = {}
        
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/tools/{tool_name}/call",
                json={"arguments": arguments}
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract text from response
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0].get("text", str(result))
                else:
                    return str(result)
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
    except httpx.ConnectError:
        raise Exception(f"Cannot connect to MCP server at {base_url}")
    except Exception as e:
        raise Exception(f"MCP client error: {str(e)}")

async def main():
    print("🔧 Starting MCP HTTP Client...")
    
    # List available tools
    tools = [
        ("chat_gemma3", "Chat with Gemma3 via Ollama"),
        ("fan_in_analysis", "Analyze transaction graph for fan-in patterns")
    ]
    
    print("Available tools:")
    for name, description in tools:
        print(f"  - {name}: {description}")
    
    print("\nCommands:")
    print("  'chat: <msg>' - Chat with Gemma3")
    print("  'fanin' - Run fan-in analysis")
    print("  'fanin: uri user pass' - Run with custom Neo4j credentials")
    print("  'quit' - Exit")
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'quit':
            break
                        
        elif user_input.startswith('chat:'):
            msg = user_input[5:].strip()
            try:
                result = await call_mcp_tool("chat_gemma3", {"message": msg})
                print(result)
            except Exception as e:
                print(f"Error: {e}")
        
        elif user_input.lower() == 'fanin':
            print("🚀 Running fan-in analysis...")
            try:
                result = await call_mcp_tool("fan_in_analysis", {})
                print(result)
            except Exception as e:
                print(f"Error: {e}")
        
        elif user_input.startswith('fanin:'):
            # Parse custom credentials: fanin: uri user pass
            parts = user_input[6:].strip().split()
            if len(parts) >= 3:
                uri, user, password = parts[0], parts[1], parts[2]
                output_file = parts[3] if len(parts) > 3 else None
                print(f"🚀 Running fan-in analysis with custom credentials...")
                try:
                    params = {
                        "neo4j_uri": uri,
                        "neo4j_user": user, 
                        "neo4j_password": password
                    }
                    if output_file:
                        params["output_file"] = output_file
                    
                    result = await call_mcp_tool("fan_in_analysis", params)
                    print(result)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Usage: fanin: <uri> <user> <password> [output_file]")
        
        else:
            print("Unknown command. Try 'chat:', 'fanin', or 'quit'")


if __name__ == "__main__":
    asyncio.run(main())