"""
FastAPI app converted to MCP Server with Fan-In Analysis 
Now using HTTP transport for Docker compatibility
"""

import math
import requests
import os
import pandas as pd
import numpy as np
import glob
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from fanin_standalone import standalone_fan_in_analysis

@tool
def csv_feature_analysis(csv_filename: str = None) -> str:
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
def exploratory_data_analysis(csv_filename: str = None) -> str:
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

**🎯 Next Steps Recommendations:**
• Use feature engineering for highly skewed variables
• Investigate high correlations for multicollinearity
• Handle missing data using appropriate imputation
• Consider outlier treatment for affected variables
• Explore categorical encoding for ML models

**🔗 Interactive Analysis:**
[Open Feature Analyzer](http://localhost:7860) for detailed visualizations and deeper analysis."""

        return eda_report
        
    except Exception as e:
        return f"❌ Error in EDA: {str(e)}"

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
csv_analysis_tool = to_fastmcp(csv_feature_analysis)
eda_tool = to_fastmcp(exploratory_data_analysis)
fanin_tool = to_fastmcp(fan_in_analysis)
chat_tool = to_fastmcp(chat_gemma3)

# Create MCP server
mcp = FastMCP(name="Money Laundering Analysis MCP Server", 
              tools=[csv_analysis_tool, eda_tool, fanin_tool, chat_tool] 
            )

if __name__ == "__main__":
    print("🚀 Starting Money Laundering Analysis MCP Server with HTTP transport")
    print(f"Neo4j URI: {os.getenv('NEO4J_URI', 'Not set')}")
    print(f"Neo4j User: {os.getenv('NEO4J_USERNAME', 'Not set')}")
    print(f"Ollama host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print("\nAvailable tools:")
    print("- csv_feature_analysis: Analyze CSV features with Gradio app link")
    print("- exploratory_data_analysis: Comprehensive EDA with insights and recommendations")
    print("- fan_in_analysis: Perform fan-in analysis on Neo4j graph")
    print("- chat_gemma3: Chat with Gemma3 via Ollama")
    print("🔧 Debug mode enabled - will show which tools are called")
    print("🌐 MCP Server will be available on http://0.0.0.0:8000")
    
    # Use the correct FastMCP run method
    mcp.run()