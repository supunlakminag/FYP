import os

folders = [
    "data/raw",
    "data/processed",
    "preprocessing",
    "model",
    "rolling_window",
    "signal_generation",
    "backtesting",
    "results"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}")

# Create empty __init__.py files so Python treats directories as packages
for folder in ["preprocessing", "model", "rolling_window", "signal_generation", "backtesting"]:
    with open(f"{folder}/__init__.py", "w") as f:
        pass