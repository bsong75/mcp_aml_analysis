"""
Standalone Fan-In Analysis Module  
Easy import to the MCP tool
"""
import os
import pandas as pd
from datetime import datetime
from graphdatascience import GraphDataScience

class FanInAnalyzer:
    def __init__(self, gds_instance):
        self.gds = gds_instance
        print("🔧 FanInAnalyzer initialized")
    
    def perform_fan_in_analysis(self):
        """Perform fan-in analysis to identify accounts receiving from many sources"""
        print("\n=== FAN-IN ANALYSIS ===")
        
        # Calculate fan-in using direct transaction analysis
        fan_in_query = """
                    MATCH (t:Transaction)
                    WITH t.to_account as target_account, 
                        count(DISTINCT t.from_account) as unique_senders,
                        count(t) as total_incoming_transactions,
                        sum(t.amount) as total_incoming_amount,
                        collect(DISTINCT t.from_account) as sender_accounts
                    MATCH (target:Account {account_id: target_account})
                    RETURN target_account as account_id,
                        unique_senders,
                        total_incoming_transactions,
                        total_incoming_amount,
                        sender_accounts
                    ORDER BY unique_senders DESC, total_incoming_amount DESC
                    """
        
        print("🔍 Executing fan-in query...")
        fan_in_results = self.gds.run_cypher(fan_in_query)
        print(f"📊 Found {len(fan_in_results)} accounts with incoming transactions")
        
        # Print comprehensive results
        print(f"\nTop {min(10, len(fan_in_results))} accounts with highest fan-in:")
        print("-" * 140)
        print(f"{'Rank':<5} {'Account ID':<15} {'Unique Senders':<15} {'Total Transactions':<18} {'Total Amount':<15}")
        print("-" * 140)
        
        for idx, row in fan_in_results.head(10).iterrows():
            rank = idx + 1
            account_id = str(row['account_id'])[:14]
            unique_senders = row['unique_senders']
            total_transactions = row['total_incoming_transactions']
            total_amount = f"${row['total_incoming_amount']:,.2f}"
            
            print(f"{rank:<5} {account_id:<15} {unique_senders:<15} {total_transactions:<18} {total_amount:<15}")
        
        return fan_in_results
    
    def identify_high_risk_accounts(self, fan_in_results):
        """Identify and analyze high-risk accounts based on fan-in patterns"""
        print("🚨 Identifying high-risk accounts...")
        
        # Identify high-risk accounts
        high_risk_threshold = fan_in_results['unique_senders'].quantile(0.9)  # Top 10%
        high_risk_accounts = fan_in_results[fan_in_results['unique_senders'] >= high_risk_threshold]
        
        print(f"\n=== HIGH-RISK ACCOUNTS (Fan-in >= {high_risk_threshold:.0f} unique senders) ===")
        print(f"Found {len(high_risk_accounts)} high-risk accounts:")
        
        for idx, row in high_risk_accounts.head(10).iterrows():
            account_id = row['account_id']
            unique_senders = row['unique_senders']
            total_amount = row['total_incoming_amount']
            
            print(f"• Account {account_id}: {unique_senders} unique senders, ${total_amount:,.2f} total")
            
            # Check if this account is actually involved in laundering
            actual_laundering = self.check_account_laundering_activity(account_id)
            if actual_laundering > 0:
                print(f"  ⚠️  CONFIRMED: This account has {actual_laundering} known laundering transactions")
            else:
                print(f"  ✓ No confirmed laundering activity in dataset")
        
        return high_risk_accounts
    
    def check_account_laundering_activity(self, account_id):
        """Check if an account is involved in known laundering transactions"""
        laundering_query = """
        MATCH (a:Account {account_id: $account_id})
        MATCH (t:Transaction)
        WHERE (t.from_account = $account_id OR t.to_account = $account_id) 
        AND t.is_laundering = 1
        RETURN count(t) as laundering_count
        """
        
        try:
            result = self.gds.run_cypher(laundering_query, {'account_id': str(account_id)})
            return result.iloc[0]['laundering_count'] if len(result) > 0 else 0
        except Exception as e:
            print(f"Warning: Could not check laundering activity for {account_id}: {e}")
            return 0
    
    def generate_statistics(self, fan_in_results):
        """Generate comprehensive fan-in statistics"""
        print(f"\n=== FAN-IN STATISTICS ===")
        print(f"Total accounts analyzed: {len(fan_in_results)}")
        print(f"Average unique senders per account: {fan_in_results['unique_senders'].mean():.2f}")
        print(f"Median unique senders per account: {fan_in_results['unique_senders'].median():.2f}")
        print(f"Max unique senders for any account: {fan_in_results['unique_senders'].max()}")
        print(f"Accounts with fan-in > 5: {len(fan_in_results[fan_in_results['unique_senders'] > 5])}")
        print(f"Accounts with fan-in > 10: {len(fan_in_results[fan_in_results['unique_senders'] > 10])}")
        print(f"Accounts with fan-in > 20: {len(fan_in_results[fan_in_results['unique_senders'] > 20])}")
        
        # Distribution analysis
        percentiles = [50, 75, 90, 95, 99]
        print(f"\nFan-in distribution percentiles:")
        for p in percentiles:
            value = fan_in_results['unique_senders'].quantile(p/100)
            print(f"  {p}th percentile: {value:.1f} unique senders")
    
    def save_results(self, fan_in_results, filename=None):
        """Save fan-in analysis results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'fan_in_analysis_results_{timestamp}.csv'
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        os.makedirs(directory, exist_ok=True)
        
        fan_in_results.to_csv(filename, index=False)
        print(f"\n💾 Fan-in analysis results saved to '{filename}'")
        return filename
    
    def run_complete_analysis(self):
        """
        Run the complete fan-in analysis pipeline
        Returns:
            dict: Dictionary containing all analysis results
        """
        print(" === STARTING COMPLETE FAN-IN ANALYSIS ===")
        
        try:
            # Step 1: Perform basic fan-in analysis
            fan_in_results = self.perform_fan_in_analysis()
            
            # Step 2: Identify high-risk accounts
            high_risk_accounts = self.identify_high_risk_accounts(fan_in_results)
            
            # Step 3: Generate statistics
            self.generate_statistics(fan_in_results)
            
            # Step 4: Save results
            results_file = self.save_results(fan_in_results)
            
            print("✅ === FAN-IN ANALYSIS COMPLETE ===")
            
            # Return all results for further processing
            return {
                'fan_in_results': fan_in_results,
                'high_risk_accounts': high_risk_accounts,
                'results_file': results_file
            }
        except Exception as e:
            print(f"❌ Error during fan-in analysis: {e}")
            raise

def standalone_fan_in_analysis(neo4j_uri=None, neo4j_user=None, neo4j_password=None, output_file=None):
    """
    Standalone fan-in analysis function that doesn't require MCP
    Args:
        neo4j info
        output_file: Optional custom output file path for results CSV
    Returns:
        String with analysis summary and file location or error message
    """
    print("\n ============  STANDALONE FAN-IN ANALYSIS CALLED!  ================")

    try:
        # Get Neo4j connection details from environment variables if not provided
        uri = neo4j_uri or os.environ.get('NEO4J_URI')
        user = neo4j_user or os.environ.get('NEO4J_USERNAME')
        password = neo4j_password or os.environ.get('NEO4J_PASSWORD')
        
        # Connect to Neo4j
        print(f"🔌 Connecting to Neo4j at {uri}...")
        gds = GraphDataScience(uri, auth=(user, password))
        print("✅ Connected to Neo4j successfully")
        
        # Check if database has data
        print("🔍 Checking database content...")
        node_count_result = gds.run_cypher("MATCH (n) RETURN count(n) as count")
        node_count = node_count_result.iloc[0]['count'] if len(node_count_result) > 0 else 0
                
        print(f"📊 Connected to Neo4j. Found {node_count} nodes in database.")
        
        # Check for Transaction nodes specifically
        tx_count_result = gds.run_cypher("MATCH (t:Transaction) RETURN count(t) as count")
        tx_count = tx_count_result.iloc[0]['count'] if len(tx_count_result) > 0 else 0
        print(f"💳 Found {tx_count} transaction nodes")
        
        # Initialize and run fan-in analyzer
        print("🔧 Initializing fan-in analyzer...")
        analyzer = FanInAnalyzer(gds)
        results = analyzer.run_complete_analysis()
        
        # Close connection
        print("🔌 Closing Neo4j connection...")
        gds.close()
        
        # Prepare summary
        fan_in_results = results['fan_in_results']
        high_risk_accounts = results['high_risk_accounts']
        results_file = results['results_file']
        
        summary = f"""✅ Fan-In Analysis Complete!

                📊 Summary:
                - Total accounts analyzed: {len(fan_in_results)}
                - High-risk accounts identified: {len(high_risk_accounts)}
                - Average unique senders per account: {fan_in_results['unique_senders'].mean():.2f}
                - Maximum fan-in detected: {fan_in_results['unique_senders'].max()} unique senders
                - Results saved to: {results_file}

                        🚨 Top 3 highest fan-in accounts:
                    """
        
        for idx, row in fan_in_results.head(3).iterrows():
            summary += f"\n• Account {row['account_id']}: {row['unique_senders']} senders, ${row['total_incoming_amount']:,.2f} total"
        
        print("🎉 Fan-in analysis completed successfully!")
        return summary
        
    except Exception as e:
        error_msg = f"❌ Error during fan-in analysis: {str(e)}"
        print(error_msg)
        return error_msg