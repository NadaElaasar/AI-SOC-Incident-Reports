# AI-Powered Cybersecurity SOC Analyst Simulator
**With Automated Reports and Embedded Charts**

---

##  Overview
This project simulates a **Security Operations Center (SOC) analyst** using AI and Machine Learning.  
It analyzes authentication logs to detect **suspicious login attempts**, potential **brute force attacks**, and other anomalous behaviors.  

The system generates:

- **Text-based incident reports**  
- **HTML reports with interactive charts**  
- **Machine Learning analysis (Isolation Forest)**  

---

##  Features

1. **Data Processing**
   - Handles large CSV files efficiently with optional sampling
   - Standardizes column names (`timestamp`, `ip`, `login_success`)
   - Supports different dataset structures for login logs

2. **Security Analysis**
   - Detects failed login attempts
   - Flags suspicious IP addresses based on thresholds
   - Provides brute-force attack detection

3. **Machine Learning**
   - Uses **Isolation Forest** for anomaly detection
   - Labels IPs as `Suspicious` or `Normal`
   - Highlights anomalous login patterns

4. **Charts & Visualizations**
   - Top IP attackers (bar chart)
   - Success vs Failure distribution (pie chart)
   - Login timeline over time (line chart)
   - Charts are embedded in the HTML report using base64 encoding

5. **Professional Reporting**
   - HTML report with charts, tables, and risk assessment
   - Text report for quick review
   - Recommendations for immediate, short-term, and long-term actions

---

##  Project Structure


AI-SOC-Report/
├── run_report.py # Main script to run the analysis
├── your_dataset.csv # Input dataset (example)
├── reports/ # Folder where reports will be saved
├── requirements.txt # Python dependencies
└── README.md # Project description and instructions


---

##  Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/AI-SOC-Report.git
cd AI-SOC-Report
Install dependencies
pip install -r requirements.txt

2. **The data**
For testing, you can use any publicly available authentication log datasets, e.g.:

- [Kaggle RBA Dataset](https://www.kaggle.com/datasets/rba/realistic-banking-authentication)

