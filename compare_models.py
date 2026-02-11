import pandas as pd
from preprocessing.data_loader import DataLoader
from model.baseline_runner import BaselineRunner
from model.intermediate_runners import IntermediateRunner # NEW IMPORT
from rolling_window.roll_train import RollingWindowTrainer

# 1. Setup
TICKER = "BTC-USD"
print(f"--- STARTING FULL ABLATION STUDY FOR {TICKER} ---")

# 2. Get Data
loader = DataLoader(TICKER)
df = loader.get_raw_data() 

if df is not None:
    results_list = []

    # --- B0: Baseline (Raw / No Scaling) ---
    b0 = BaselineRunner(df)
    results_list.append(b0.run())

    # ---  B1.1: Drop Missing + Global Scaling (Static) ---
    b1_1 = IntermediateRunner(df, mode='drop')
    results_list.append(b1_1.run())

    # ---  B1.2: Interpolate + Global Scaling (Static) ---
    b1_2 = IntermediateRunner(df, mode='interpolate')
    results_list.append(b1_2.run())

    # ---  Final: Proposed Evaluation (Rolling + Dynamic) ---
    eval_model = RollingWindowTrainer(df, window_size=168, debug_mode=True)
    results_list.append(eval_model.run())

    # 5. Compile Report
    df_report = pd.DataFrame(results_list)
    
    # Reorder Columns
    cols = [
        "Model_Type", 
        "Preprocessing", 
        "Normalization", 
        "Training_Strategy", 
        "Novelty_Contribution",
        "RMSE (Error Cost)", 
        "MAPE (Error %)"
    ]
    df_report = df_report[cols]

    # 6. Save
    output_path = "results/Final_Detailed_Comparison.xlsx"
    df_report.to_excel(output_path, index=False)
    
    print("\n✅ FULL EXPERIMENT REPORT GENERATED")
    print(f"📊 Saved to: {output_path}")
    print("-" * 30)
    print(df_report)