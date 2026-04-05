# ==============================================
# AI-Powered Cybersecurity System - SOC Analyst Simulator
# WITH CHARTS INSIDE HTML REPORT
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import os
import gc
import base64
from io import BytesIO

# ==============================================
# 1 Data Processing - FOR RBA DATASET
# ==============================================

def load_and_preprocess_data_optimized(file_path, sample_fraction=0.05, max_rows=300000):
    """Load large CSV files efficiently"""
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    if file_size > 100:
        print(f"Large file detected. Using sampling ({sample_fraction*100}%)...")
        
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1
        print(f"Total rows in file: {total_lines:,}")
        
        skip_rows = np.random.choice(range(1, total_lines+1), 
                                      size=int(total_lines * (1 - sample_fraction)), 
                                      replace=False)
        
        df = pd.read_csv(file_path, skiprows=skip_rows, low_memory=False)
        
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
            print(f"Sampled down to {max_rows:,} rows")
    else:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except MemoryError:
            print("Memory error. Loading in chunks...")
            chunks = []
            chunk_size = 50000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= max_rows:
                    break
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
    
    print(f"Loaded {len(df):,} rows")
    
    # Rename columns to standard names
    df.columns = df.columns.str.lower()
    
    # Handle timestamp column
    if 'login timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['login timestamp'], errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Handle IP column
    ip_col = None
    if 'ip address' in df.columns:
        ip_col = 'ip address'
    elif 'ip' in df.columns:
        ip_col = 'ip'
    elif 'ip_address' in df.columns:
        ip_col = 'ip_address'
    elif 'src_ip' in df.columns:
        ip_col = 'src_ip'
    
    # Handle login success column
    if 'login successful' in df.columns:
        df['login_success'] = df['login successful'].apply(
            lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'success'] else 0
        )
    elif 'status' in df.columns:
        df['login_success'] = df['status'].apply(
            lambda x: 1 if str(x).lower() in ['success', 'ok', '200', 'true'] else 0
        )
    elif 'login_success' in df.columns:
        df['login_success'] = df['login_success'].apply(
            lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'success'] else 0
        )
    else:
        print("Warning: No login success column found. Creating default.")
        df['login_success'] = 1
    
    # Drop rows with missing IP
    if ip_col:
        df = df.dropna(subset=[ip_col])
        print(f"IP column used: {ip_col}")
    
    print(f"Login success column processed")
    print(f"Final dataset size: {len(df):,} rows")
    
    return df, ip_col

# ==============================================
# 2 Security Analysis
# ==============================================

def analyze_security_events(df, ip_col):
    """Detect brute force and suspicious patterns"""
    if ip_col not in df.columns or ip_col is None:
        return None, None
    
    failed_attempts = df[df['login_success'] == 0].groupby(ip_col).size().reset_index(name='failed_count')
    total_attempts = df.groupby(ip_col).size().reset_index(name='total_count')
    
    ip_stats = total_attempts.merge(failed_attempts, on=ip_col, how='left').fillna(0)
    ip_stats['failed_ratio'] = ip_stats['failed_count'] / ip_stats['total_count']
    ip_stats['brute_force_suspect'] = (ip_stats['failed_count'] > 10) | (ip_stats['failed_ratio'] > 0.8)
    ip_stats = ip_stats.sort_values('failed_count', ascending=False)
    
    return ip_stats, failed_attempts

# ==============================================
# 3 Machine Learning (Isolation Forest)
# ==============================================

def detect_anomalies_ml(df, ip_col):
    """Use Isolation Forest to detect anomalous IP behavior"""
    if ip_col is None or ip_col not in df.columns:
        return None
    
    ip_features = df.groupby(ip_col).agg(
        total_attempts=('login_success', 'count'),
        failed_attempts=('login_success', lambda x: (x == 0).sum()),
        success_rate=('login_success', 'mean')
    ).reset_index()
    
    X = ip_features[['total_attempts', 'failed_attempts', 'success_rate']].fillna(0)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    ip_features['anomaly'] = iso_forest.fit_predict(X)
    ip_features['anomaly_label'] = ip_features['anomaly'].apply(lambda x: 'Suspicious' if x == -1 else 'Normal')
    ip_features = ip_features.sort_values('failed_attempts', ascending=False)
    
    return ip_features

# ==============================================
# 4 Generate Charts as Base64 for HTML
# ==============================================

def generate_charts_as_base64(df, ip_col):
    """Generate charts and return as base64 strings for embedding in HTML"""
    
    charts = {}
    
    if ip_col is None or ip_col not in df.columns:
        return charts
    
    # Chart 1: Top Attackers Bar Chart
    try:
        ip_summary = df.groupby(ip_col)['login_success'].value_counts().unstack(fill_value=0)
        
        if 0 in ip_summary.columns:
            top_failed = ip_summary[0].sort_values(ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            colors = ['#d32f2f' if x > 20 else '#ff9800' if x > 10 else '#ffeb3b' for x in top_failed.values]
            bars = ax.bar(range(len(top_failed)), top_failed.values, color=colors)
            ax.set_xticks(range(len(top_failed)))
            ax.set_xticklabels(top_failed.index, rotation=45, ha='right', fontsize=9)
            ax.set_xlabel('IP Address', fontsize=12)
            ax.set_ylabel('Number of Failed Attempts', fontsize=12)
            ax.set_title('Top 15 IP Addresses by Failed Login Attempts', fontsize=14, fontweight='bold')
            ax.set_facecolor('#f9f9f9')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, top_failed.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            charts['top_attackers'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
    except Exception as e:
        print(f"Could not generate chart 1: {e}")
    
    # Chart 2: Success vs Failure Pie Chart
    try:
        success_count = (df['login_success'] == 1).sum()
        failure_count = (df['login_success'] == 0).sum()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#4caf50', '#d32f2f']
        labels = [f'Successful\n({success_count:,})', f'Failed\n({failure_count:,})']
        sizes = [success_count, failure_count]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        ax.set_title('Login Attempts: Success vs Failure', fontsize=14, fontweight='bold')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        charts['pie_chart'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
    except Exception as e:
        print(f"Could not generate chart 2: {e}")
    
    # Chart 3: Time Series Line Chart
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        try:
            df['hour'] = df['timestamp'].dt.floor('H')
            hourly = df.groupby(['hour', 'login_success']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            if 0 in hourly.columns:
                ax.plot(hourly.index, hourly[0], color='#d32f2f', linewidth=2.5, 
                       label='Failed Attempts', marker='o', markersize=4)
            if 1 in hourly.columns:
                ax.plot(hourly.index, hourly[1], color='#4caf50', linewidth=2.5, 
                       label='Successful Attempts', marker='s', markersize=4)
            
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Number of Attempts', fontsize=12)
            ax.set_title('Login Attempts Over Time', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f9f9f9')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            charts['timeline'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print(f"Could not generate chart 3: {e}")
    
    return charts

# ==============================================
# 5 Generate Professional Report with Embedded Charts
# ==============================================

def generate_professional_report(ip_stats, ml_results, df, ip_col, charts):
    """Generate a professional incident report as text and HTML with embedded charts"""
    
    if ml_results is None:
        return "Error: Could not generate report due to missing data.", ""
    
    suspicious_ips = ml_results[ml_results['anomaly_label'] == 'Suspicious'][ip_col].tolist() if ip_col else []
    brute_force_ips = ip_stats[ip_stats['brute_force_suspect']][ip_col].tolist() if ip_stats is not None and ip_col else []
    
    # Get top attackers
    top_attackers = ip_stats.head(10) if ip_stats is not None else []
    
    # Time analysis
    time_window = None
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        time_window = {
            'start': df_sorted['timestamp'].min(),
            'end': df_sorted['timestamp'].max()
        }
    
    total_failures = (df['login_success'] == 0).sum()
    total_attempts = len(df)
    unique_ips = df[ip_col].nunique() if ip_col else 0
    failure_rate = (total_failures/total_attempts*100) if total_attempts > 0 else 0
    
    # Generate HTML report with embedded charts
    html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Security Incident Report - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }}
        .section h2 {{
            color: #d32f2f;
            margin-bottom: 20px;
            font-size: 24px;
            border-left: 4px solid #d32f2f;
            padding-left: 15px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 36px;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 14px;
            margin-top: 10px;
            opacity: 0.9;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .risk-critical {{
            background-color: #d32f2f;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .recommendation-box {{
            background-color: #fff3e0;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #ff9800;
            border-radius: 5px;
        }}
        .recommendation-box strong {{
            color: #e65100;
        }}
        .footer {{
            background-color: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .badge-critical {{
            background-color: #d32f2f;
            color: white;
        }}
        .badge-warning {{
            background-color: #ff9800;
            color: white;
        }}
        .badge-monitor {{
            background-color: #2196f3;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>INCIDENT RESPONSE REPORT</h1>
            <p>Generated by AI-Powered SOC Analyst System</p>
            <p><strong>Report ID:</strong> IR-{datetime.now().strftime('%Y%m%d-%H%M%S')}</p>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Severity:</strong> <span style="background:#ff9800;padding:5px 15px;border-radius:20px;">CRITICAL</span></p>
        </div>
        
        <div class="content">
            <!-- Section 1: Executive Summary -->
            <div class="section">
                <h2>EXECUTIVE SUMMARY</h2>
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-number">{total_attempts:,}</div><div class="stat-label">Total Events</div></div>
                    <div class="stat-card"><div class="stat-number">{unique_ips:,}</div><div class="stat-label">Unique IPs</div></div>
                    <div class="stat-card"><div class="stat-number">{total_failures:,}</div><div class="stat-label">Failed Logins</div></div>
                    <div class="stat-card"><div class="stat-number">{failure_rate:.1f}%</div><div class="stat-label">Failure Rate</div></div>
                </div>
                <p style="margin-top:20px;line-height:1.6;">This report presents an automated security analysis of authentication logs. The system detected suspicious activities indicating a potential brute force attack or credential stuffing campaign targeting the organization's authentication infrastructure.</p>
            </div>
"""
    
    # Add charts if they exist
    if 'top_attackers' in charts:
        html_report += f"""
            <div class="section">
                <h2>TOP ATTACKERS ANALYSIS</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts['top_attackers']}" alt="Top Attackers Chart">
                </div>
            </div>
        """
    
    if 'pie_chart' in charts:
        html_report += f"""
            <div class="section">
                <h2>SUCCESS vs FAILURE DISTRIBUTION</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts['pie_chart']}" alt="Success Failure Distribution">
                </div>
            </div>
        """
    
    if 'timeline' in charts:
        html_report += f"""
            <div class="section">
                <h2>TIMELINE ANALYSIS</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts['timeline']}" alt="Timeline Analysis">
                </div>
            </div>
        """
    
    # Add top suspicious IPs table
    html_report += f"""
            <div class="section">
                <h2>TOP SUSPICIOUS IP ADDRESSES</h2>
                <table>
                    <thead>
                        <tr><th>IP Address</th><th>Total Attempts</th><th>Failed Attempts</th><th>Failure Rate</th><th>Status</th></tr>
                    </thead>
                    <tbody>
    """
    
    if len(top_attackers) > 0:
        for idx, row in top_attackers.head(15).iterrows():
            if row['failed_count'] > 20:
                status = '<span class="badge badge-critical">CRITICAL</span>'
            elif row['failed_count'] > 10:
                status = '<span class="badge badge-warning">WARNING</span>'
            else:
                status = '<span class="badge badge-monitor">MONITOR</span>'
            
            html_report += f"""
                <tr>
                    <td><strong>{row[ip_col]}</strong></td>
                    <td>{int(row['total_count'])}</td>
                    <td>{int(row['failed_count'])}</td>
                    <td>{row['failed_ratio']*100:.1f}%</td>
                    <td>{status}</td>
                </tr>
            """
    else:
        html_report += '<tr><td colspan="5">No suspicious IPs detected</td></tr>'
    
    html_report += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>RISK ASSESSMENT</h2>
                <div class="risk-critical">
                    <strong>CRITICAL FINDINGS:</strong><br>
                    - Failure rate of {failure_rate:.2f}% indicates active credential guessing attack<br>
                    - {len(suspicious_ips)} suspicious IPs detected by ML algorithm<br>
                    - Potential account takeover (ATO) risk detected
                </div>
                <p style="margin-top:15px;"><strong>Impact Analysis:</strong> Unauthorized access to user accounts, possible data breach, service degradation.</p>
            </div>
            
            <div class="section">
                <h2>RECOMMENDATIONS</h2>
                <div class="recommendation-box"><strong>IMMEDIATE (0-4 hours):</strong> Block suspicious IPs, enable MFA, implement CAPTCHA after 3 failures</div>
                <div class="recommendation-box"><strong>SHORT-TERM (24 hours):</strong> Review failed login patterns, reset compromised passwords, increase logging</div>
                <div class="recommendation-box"><strong>LONG-TERM (1 week):</strong> Implement rate limiting, deploy WAF, update IAM policies</div>
            </div>
            
            <div class="section">
                <h2>ML DETECTION DETAILS</h2>
                <p><strong>Algorithm:</strong> Isolation Forest (Unsupervised Learning)</p>
                <p><strong>Total IPs Analyzed:</strong> {len(ml_results)}</p>
                <p><strong>Anomalous IPs Detected:</strong> {len(suspicious_ips)}</p>
                <p><strong>Contamination Factor:</strong> 0.1</p>
            </div>
        </div>
        
        <div class="footer">
            <p>This report was generated automatically by the AI-Powered SOC Analyst System.</p>
            <p>For further investigation, please contact the Security Operations Center.</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Generate TEXT report
    text_report = f"""
================================================================================
                            INCIDENT RESPONSE REPORT
================================================================================

Report ID: IR-{datetime.now().strftime('%Y%m%d-%H%M%S')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Severity Level: CRITICAL

================================================================================
1. EXECUTIVE SUMMARY
================================================================================

Total events analyzed: {total_attempts:,}
Unique IP addresses: {unique_ips:,}
Failed login attempts: {total_failures:,}
Failure rate: {failure_rate:.2f}%
Suspicious IPs detected: {len(suspicious_ips)}
Brute force suspects: {len(brute_force_ips)}

================================================================================
2. TOP SUSPICIOUS IP ADDRESSES
================================================================================
"""
    
    if len(top_attackers) > 0:
        for idx, row in top_attackers.head(15).iterrows():
            text_report += f"\n{row[ip_col]}: {int(row['failed_count'])} failures out of {int(row['total_count'])} attempts ({row['failed_ratio']*100:.1f}%)"
    else:
        text_report += "\nNo suspicious IPs detected"
    
    text_report += f"""

================================================================================
3. RECOMMENDATIONS
================================================================================

IMMEDIATE: Block suspicious IPs, enable MFA, implement CAPTCHA
SHORT-TERM: Review patterns, reset compromised passwords
LONG-TERM: Rate limiting, WAF deployment, policy review

================================================================================
                            END OF REPORT
================================================================================
"""
    
    return text_report, html_report

# ==============================================
# 6 Save Reports
# ==============================================

def save_reports(text_report, html_report, output_dir='reports'):
    """Save all reports to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text report
    text_path = f'{output_dir}/incident_report.txt'
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_report)
    print(f"Text report saved: {text_path}")
    
    # Save HTML report
    html_path = f'{output_dir}/incident_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"HTML report saved: {html_path}")
    
    print(f"\nReports saved in: {output_dir}/")
    print(f"  - Open incident_report.html in your browser for a formatted report with charts")

# ==============================================
# MAIN EXECUTION PIPELINE
# ==============================================

def run_cybersecurity_pipeline(csv_file_path, sample_fraction=0.05):
    """Run the full SOC analyst simulation"""
    print("\n" + "="*60)
    print("AI CYBERSECURITY SYSTEM - SOC ANALYST SIMULATOR")
    print("="*60 + "\n")
    
    print("STEP 1: Loading and preprocessing data...")
    df, ip_col = load_and_preprocess_data_optimized(csv_file_path, sample_fraction)
    
    if ip_col is None:
        print("Error: No IP column found in dataset.")
        print(f"Available columns: {list(df.columns)}")
        return None, None
    
    print("\nSTEP 2: Analyzing security events...")
    ip_stats, failed_attempts = analyze_security_events(df, ip_col)
    
    print("\nSTEP 3: Running ML anomaly detection (Isolation Forest)...")
    ml_results = detect_anomalies_ml(df, ip_col)
    
    if ml_results is None:
        print("Error: ML detection failed.")
        return None, None
    
    print("\nSTEP 4: Generating charts...")
    charts = generate_charts_as_base64(df, ip_col)
    print(f"Generated {len(charts)} charts")
    
    print("\nSTEP 5: Generating professional incident report...")
    text_report, html_report = generate_professional_report(ip_stats, ml_results, df, ip_col, charts)
    
    print("\nSTEP 6: Saving reports...")
    save_reports(text_report, html_report)
    
    # Print summary to console
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total events analyzed: {len(df):,}")
    print(f"Failed attempts: {(df['login_success'] == 0).sum():,}")
    print(f"Suspicious IPs found: {len(ml_results[ml_results['anomaly_label'] == 'Suspicious'])}")
    print(f"Charts generated: {len(charts)}")
    print("="*60)
    
    return ml_results, text_report

# ==============================================
# RUN THE PROJECT
# ==============================================

if __name__ == "__main__":
    
    FOLDER_PATH = r"C:\Users\elaas\Desktop\AI+security\archive (1)"
    
    csv_files = []
    if os.path.exists(FOLDER_PATH):
        for file in os.listdir(FOLDER_PATH):
            if file.endswith('.csv'):
                csv_files.append(file)
    
    if csv_files:
        print(f"Found {len(csv_files)} CSV file(s):")
        for i, file in enumerate(csv_files):
            print(f"   {i+1}. {file}")
        
        selected_file = csv_files[0]
        csv_path = os.path.join(FOLDER_PATH, selected_file)
        print(f"\nSelected: {selected_file}")
        
        # Run analysis with 5% sample
        results, report = run_cybersecurity_pipeline(csv_path, sample_fraction=0.05)
        
        if results is not None:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\nTo view the report with charts:")
            print("  Open 'reports/incident_report.html' in your web browser")
            print("\nThe report now includes:")
            print("  - Bar chart of top attackers")
            print("  - Pie chart of success vs failure")
            print("  - Timeline chart of attacks")
            print("  - All charts embedded directly in the HTML")
    else:
        print(f"No CSV files found in: {FOLDER_PATH}")