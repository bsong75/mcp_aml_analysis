"""
CBP Agriculture Analysis MCP Server
Provides tools for CSV analysis, EDA, Neo4j visualization, chat, and acronym lookup
"""

import math
import requests
import os
import pandas as pd
import numpy as np
import glob
import json
from typing import Optional
from difflib import get_close_matches
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
import uvicorn

# Load CBP Acronyms Database
def load_acronyms():
    """Load acronyms from JSON file"""
    acronym_file = os.path.join(os.path.dirname(__file__), 'cbp_acronyms.json')
    try:
        with open(acronym_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Warning: {acronym_file} not found. Using empty acronym database.")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️ Warning: Error parsing {acronym_file}: {e}")
        return {}

def save_acronyms(acronyms_dict):
    """Save acronyms to JSON file with alphabetical sorting"""
    acronym_file = os.path.join(os.path.dirname(__file__), 'cbp_acronyms.json')
    try:
        # Sort by keys (acronyms) alphabetically
        sorted_acronyms = dict(sorted(acronyms_dict.items()))
        with open(acronym_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_acronyms, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"⚠️ Error saving {acronym_file}: {e}")
        return False

CBP_ACRONYMS = load_acronyms()

@tool
def csv_feature_analysis(csv_filename: Optional[str] = None) -> str:
    """Analyze CSV features and provide summary with Gradio app link
    Args:
        csv_filename: Optional specific CSV file name in graph_features_files folder
    Returns:
        String with analysis summary and Gradio app link
    """
    print("📊 CSV FEATURE ANALYSIS TOOL CALLED!")
    
    try:
        # Define the features folder
        features_folder = "/app/graph_features_files"
        
        # If no specific file provided, find available CSV files
        if not csv_filename:
            csv_files = glob.glob(f"{features_folder}/*.csv")
            if not csv_files:
                return "❌ No CSV files found in graph_features_files folder"
            csv_file = csv_files[0]  # Use first available CSV
            csv_filename = os.path.basename(csv_file)
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
        summary = f"""✅ **Analysis of Features Complete**
        
                    **File:** {csv_filename}
                    **Dataset Size:** {rows} rows × {cols} columns
                    **Features Analyzed:** {len(feature_cols)} numeric features
                            
                    **Key Insights:**
                    • Total features: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}
                    • Most variable features: {', '.join(top_variable) if top_variable else 'None'}
                    • Data completeness: {100 - (df.isnull().sum().sum() / (rows * cols) * 100):.1f}%

                    🎯 **Interactive Analysis Available:**
                    [Open Feature Analyzer](http://localhost:7860)                   
                    """
        
        return summary
        
    except Exception as e:
        return f"❌ Error analyzing CSV: {str(e)}"

@tool
def exploratory_data_analysis(csv_filename: Optional[str] = None) -> str:
    """Perform comprehensive Exploratory Data Analysis on CSV file
    Args:
        csv_filename: Optional specific CSV file name in graph_features_files folder
    Returns:
        String with comprehensive EDA summary and insights
    """
    print("🔍 EXPLORATORY DATA ANALYSIS TOOL CALLED!")
    
    try:
        # Define the features folder
        features_folder = "/app/graph_features_files"
        
        # If no specific file provided, find available CSV files
        if not csv_filename:
            csv_files = glob.glob(f"{features_folder}/*.csv")
            if not csv_files:
                return "❌ No CSV files found in graph_features_files folder"
            csv_file = csv_files[0]  # Use first available CSV
            csv_filename = os.path.basename(csv_file)
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
[Open EDA Dashboard](http://localhost:8001/dashboard?file={csv_filename}) for detailed visualizations and deeper analysis."""

        return eda_report
        
    except Exception as e:
        return f"❌ Error in EDA: {str(e)}"

@tool
def neo4j_visualization() -> str:
    """Get Neo4j graph database summary and visualization link
    Returns:
        String with graph statistics and Neo4j browser link
    """
    print("🔍 NEO4J VISUALIZATION TOOL CALLED!")

    try:
        from neo4j import GraphDatabase

        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://host.docker.internal:7687')
        neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

        print(f"Connecting to Neo4j at {neo4j_uri}...")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print("Driver created successfully")

        with driver.session() as session:
            # Get node count
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()['count']

            # Get relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()['count']

            # Get node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record['label'] for record in labels_result]

            # Get relationship types
            types_result = session.run("CALL db.relationshipTypes()")
            rel_types = [record['relationshipType'] for record in types_result]

        driver.close()

        summary = f"""📊 **Neo4j Graph Database Summary:**

**Statistics:**
• Total Nodes: {node_count:,}
• Total Relationships: {rel_count:,}

**Node Labels:**
{chr(10).join([f'• {label}' for label in labels])}

**Relationship Types:**
{chr(10).join([f'• {rel_type}' for rel_type in rel_types])}

🔗 **Interactive Visualization:**
• [Open Neo4j Browser](http://localhost:7474/browser/) - Explore the graph interactively"""

        return summary

    except ImportError as e:
        error_msg = f"❌ Error: neo4j-driver package not installed. Run: pip install neo4j\nDetails: {str(e)}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error connecting to Neo4j: {str(e)}\n\nMake sure Neo4j is running and credentials are correct."
        print(f"Neo4j error: {e}")
        import traceback
        traceback.print_exc()
        return error_msg

@tool
def chat_agent(message: str, model: str = "gemma3", timeout: int = 3000) -> str:
    """Chat with Gemma3 via Ollama
    Args:
        message: The message to send to the AI model
        model: The model to use (default: gemma3)
        timeout: Request timeout in seconds (default: 3000)
    Returns:
        String response from the AI model or error message
    """
    print("💬 chat_agent TOOL CALLED!")
    print(f"Message: {message}")
    print(f"Model: {model}")
    
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        url = f"{ollama_host}/v1/chat/completions"

        system_message = """
        You are ATLAS (Agentic Toolkit for Learning and Advanced Solutions), an advanced AI agent. 
        You are an expert in all matters related to Customs and Border Patrol.
        Your expertise is in importing agriculture commodities into the U.S.
        You reside in Ashburn, VA, but have visibility in all ports of entry.
        You are sophisticated, witty, efficient, and always ready to help. 
        Speak with confidence and a touch of dry humor when appropriate, 
        but remain professional and helpful. 
        You cannot do EDA on acronyms.
        You have several MCP tools to offer including:
            - Exploratory Data Analysis (EDA)
            - Graph Feature Analysis
            - Neo4j GraphDB Analysis
            - CBP Acronym helper
        When users ask for help or tools, guide them to use the appropriate commands.
        """

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
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

@tool
def acronym_lookup(acronym: str) -> str:
    """Look up CBP Agriculture acronyms with fuzzy matching
    Args:
        acronym: The acronym to look up (e.g., "AGC", "HSUSA", "RBS")
    Returns:
        Definition of the acronym or suggestions if not found
    """
    print(f"🔤 ACRONYM LOOKUP TOOL CALLED for: {acronym}")

    # Normalize input
    acronym_input = acronym.strip().upper()

    if not acronym_input:
        return "❌ Please provide an acronym to look up."

    # Exact match
    if acronym_input in CBP_ACRONYMS:
        definition = CBP_ACRONYMS[acronym_input]
        return f"✅ **{acronym_input}**: {definition}"

    # Fuzzy matching - find similar acronyms
    all_acronyms = list(CBP_ACRONYMS.keys())
    close_matches = get_close_matches(acronym_input, all_acronyms, n=3, cutoff=0.6)

    if close_matches:
        suggestions = []
        for match in close_matches:
            suggestions.append(f"  • **{match}**: {CBP_ACRONYMS[match]}")

        return f"❌ '{acronym}' not found in CBP Agriculture knowledge base.\n\n**Did you mean:**\n" + "\n".join(suggestions)
    else:
        # No close matches - show some random examples
        sample_acronyms = list(CBP_ACRONYMS.items())[:5]
        examples = [f"  • **{k}**: {v}" for k, v in sample_acronyms]

        return f"❌ '{acronym}' not found in CBP Agriculture knowledge base.\n\n**Available acronyms include:**\n" + "\n".join(examples) + f"\n\n_Total acronyms in database: {len(CBP_ACRONYMS)}_"

@tool
def acronym_update(acronym: str, definition: str) -> str:
    """Add or update a CBP Agriculture acronym in the database
    Args:
        acronym: The acronym to add/update (e.g., "FDA")
        definition: The full definition (e.g., "Food and Drug Administration")
    Returns:
        Success message or error
    """
    print(f"➕ ACRONYM UPDATE TOOL CALLED: {acronym} = {definition}")

    # Validate inputs
    acronym_clean = acronym.strip().upper()
    definition_clean = definition.strip()

    if not acronym_clean:
        return "❌ Error: Acronym cannot be empty."

    if not definition_clean:
        return "❌ Error: Definition cannot be empty."

    # Check if updating existing or adding new
    is_update = acronym_clean in CBP_ACRONYMS
    old_definition = CBP_ACRONYMS.get(acronym_clean, None)

    # Update the dictionary
    CBP_ACRONYMS[acronym_clean] = definition_clean

    # Save to file
    if save_acronyms(CBP_ACRONYMS):
        if is_update:
            return f"✅ **Acronym Updated Successfully:**\n\n**{acronym_clean}**\n• Old: {old_definition}\n• New: {definition_clean}\n\n_Total acronyms: {len(CBP_ACRONYMS)}_"
        else:
            return f"✅ **New Acronym Added:**\n\n**{acronym_clean}**: {definition_clean}\n\n_Total acronyms: {len(CBP_ACRONYMS)}_"
    else:
        return "❌ Error: Failed to save acronym to database file."

@tool
def acronym_delete(acronym: str) -> str:
    """Delete a CBP Agriculture acronym from the database
    Args:
        acronym: The acronym to delete (e.g., "FDA")
    Returns:
        Success message or error
    """
    print(f"🗑️ ACRONYM DELETE TOOL CALLED: {acronym}")

    # Normalize input
    acronym_clean = acronym.strip().upper()

    if not acronym_clean:
        return "❌ Error: Acronym cannot be empty."

    # Check if exists
    if acronym_clean not in CBP_ACRONYMS:
        # Try fuzzy matching to help user
        all_acronyms = list(CBP_ACRONYMS.keys())
        close_matches = get_close_matches(acronym_clean, all_acronyms, n=3, cutoff=0.6)

        if close_matches:
            suggestions = ", ".join([f"'{m}'" for m in close_matches])
            return f"❌ '{acronym}' not found in database.\n\n**Did you mean:** {suggestions}?"
        else:
            return f"❌ '{acronym}' not found in database."

    # Store definition before deletion
    deleted_definition = CBP_ACRONYMS[acronym_clean]

    # Delete from dictionary
    del CBP_ACRONYMS[acronym_clean]

    # Save to file
    if save_acronyms(CBP_ACRONYMS):
        return f"✅ **Acronym Deleted:**\n\n**{acronym_clean}**: {deleted_definition}\n\n_Total acronyms: {len(CBP_ACRONYMS)}_"
    else:
        # Restore if save failed
        CBP_ACRONYMS[acronym_clean] = deleted_definition
        return "❌ Error: Failed to save changes to database file."

# Convert to MCP tools
csv_analysis_tool = to_fastmcp(csv_feature_analysis)
eda_tool = to_fastmcp(exploratory_data_analysis)
neo4j_tool = to_fastmcp(neo4j_visualization)
chat_tool = to_fastmcp(chat_agent)
acronym_tool = to_fastmcp(acronym_lookup)
acronym_update_tool = to_fastmcp(acronym_update)
acronym_delete_tool = to_fastmcp(acronym_delete)

# Create MCP server
mcp = FastMCP(name="CBP Agriculture Analysis MCP Server",
              tools=[csv_analysis_tool, eda_tool, neo4j_tool, chat_tool, acronym_tool, acronym_update_tool, acronym_delete_tool]
            )

if __name__ == "__main__":
    print("🚀 Starting CBP Agriculture Analysis MCP Server with HTTP transport")
    print(f"Neo4j URI: {os.getenv('NEO4J_URI', 'Not set')}")
    print(f"Neo4j User: {os.getenv('NEO4J_USERNAME', 'Not set')}")
    print(f"Ollama host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"📚 CBP Acronyms loaded: {len(CBP_ACRONYMS)} entries")
    print("\nAvailable tools:")
    print("- csv_feature_analysis: Analyze CSV features with Gradio app link")
    print("- exploratory_data_analysis: Comprehensive EDA with insights and recommendations")
    print("- neo4j_visualization: Get Neo4j graph summary and browser link")
    print("- chat_agent: Chat with Gemma3 via Ollama")
    print("- acronym_lookup: Look up CBP Agriculture acronyms with fuzzy matching")
    print("- acronym_update: Add or update acronyms in the database")
    print("- acronym_delete: Delete acronyms from the database")
    print("🔧 Debug mode enabled - will show which tools are called")

    # Get host and port from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    print(f"🌐 MCP Server starting on http://{host}:{port}")

    # Run MCP server using the streamable_http_app with uvicorn to control host/port
    uvicorn.run(mcp.streamable_http_app, host=host, port=port)