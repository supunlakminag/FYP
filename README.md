# Bitcoin Futures Intelligence - Setup and Run Guide

This is the single setup guide for this project.

## 1. Required Software

- Python **3.10.x** (recommended: 3.10.19)
- `pip` (comes with Python)
- Internet connection (Binance/Yahoo Finance data APIs)

Use Python 3.10 for best TensorFlow compatibility on both Windows and macOS.

## 2. Project Dependencies

Core packages used by the project:

- streamlit
- pandas
- numpy
- yfinance
- ccxt
- scikit-learn
- openpyxl
- matplotlib
- seaborn
- tqdm
- blinker
- protobuf

TensorFlow differs by OS:

- macOS Apple Silicon (M1/M2/M3): `tensorflow-macos==2.13.0` + `tensorflow-metal==1.2.0`
- Windows: `tensorflow==2.13.0`

## 3. macOS Setup (Apple Silicon)

### 3.1 Install Python 3.10

```bash
brew install python@3.10
```

### 3.2 Create and activate virtual environment

```bash
cd /path/to/fyp_implement
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
python --version
```

Expected: Python 3.10.x

### 3.3 Install libraries

```bash
pip install --upgrade pip setuptools wheel
pip install tensorflow-macos==2.13.0 tensorflow-metal==1.2.0
pip install streamlit pandas numpy yfinance ccxt scikit-learn openpyxl matplotlib seaborn tqdm blinker protobuf
```

### 3.4 Create required directories

```bash
mkdir -p data/raw results model/saved_models
```

### 3.5 Verify setup

```bash
python test_setup.py
python test_api.py
python diagnose_mac.py
```

## 4. Windows Setup

### 4.1 Install Python 3.10

- Download and install Python 3.10 from `python.org`.
- During installation, enable `Add Python to PATH`.

### 4.2 Create and activate virtual environment

```bat
cd C:\path\to\fyp_implement
py -3.10 -m venv venv
venv\Scripts\activate
python --version
```

Expected: Python 3.10.x

### 4.3 Install libraries

```bat
pip install --upgrade pip setuptools wheel
pip install tensorflow==2.13.0
pip install streamlit pandas numpy yfinance ccxt scikit-learn openpyxl matplotlib seaborn tqdm blinker protobuf
```

### 4.4 Create required directories

```bat
mkdir data\raw
mkdir results
mkdir model\saved_models
```

### 4.5 Verify setup

```bat
python test_setup.py
python test_api.py
```

## 5. Run the App (Both OS)

Activate the virtual environment first, then run:

```bash
streamlit run app.py
```

Default URL:

- `http://localhost:8501`

If port 8501 is busy:

```bash
streamlit run app.py --server.port 8502
```

## 6. Quick Troubleshooting

- TensorFlow import issue on macOS: reinstall `tensorflow-macos==2.13.0` and `tensorflow-metal==1.2.0`
- TensorFlow import issue on Windows: reinstall `tensorflow==2.13.0`
- API fetch delays: first run can take longer; ensure internet is available
- Missing module: install it in the active venv with `pip install <module>`
- Disk full during install: free space and retry package installation

## 7. Notes

- This is an academic/educational project.
- Not financial advice.
# FYP
