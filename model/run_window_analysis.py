import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.data_loader import DataLoader
from rolling_window.roll_train import RollingWindowTrainer

# Set visual style
sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})

def run_analysis():
    print("🧪 STARTING LONG-TERM WINDOW SIZE SENSITIVITY ANALYSIS...")
    print("⚠️ NOTE: This will take time (Debug Mode is OFF).")
    
    # 1. Setup
    TICKER = "BTC/USDT" 
    
    # Define windows in DAYS
    days_list = [120, 180, 240, 300, 360]
    
    # Convert to HOURS (Days * 24)
    window_sizes_hours = [d * 24 for d in days_list] 
    
    results = []

    # 2. Load Data
    loader = DataLoader(TICKER)
    df = loader.get_raw_data()
    
    if df is None:
        print("❌ Data Load Failed.")
        return
        
    print(f"✅ Loaded {len(df)} rows of data. (Approx {len(df)/24:.1f} Days)")

    # 3. Loop through Window Sizes
    for days, hours in zip(days_list, window_sizes_hours):
        print(f"\n🔄 Testing Window: {days} Days ({hours} Hours)...")
        
        # Check if we have enough data for this window
        if len(df) < (hours + 500):
            print(f"⚠️ Skipping {days} Days: Not enough data loaded to train this window.")
            continue
        
        try:
            # --- CRITICAL: debug_mode=False ---
            # This ensures we run the FULL validation (The Marathon), not just 500 hours.
            trainer = RollingWindowTrainer(df, window_size=hours, debug_mode=False)
            
            # Run and capture metrics
            metrics = trainer.run()
            
            # Store result
            results.append({
                "Window_Days": days,
                "Window_Hours": hours,
                # Clean the % string to a float number for plotting
                "MAPE": float(metrics["MAPE (Error %)"].strip('%')),
                "RMSE": metrics["RMSE (Error Cost)"],
                "MAE": metrics["MAE (Avg Error $)"]
            })
            
        except Exception as e:
            print(f"⚠️ Skipping Window {days} Days due to error: {e}")
            continue

    # 4. Save Results to Excel
    if not results:
        print("❌ No results generated.")
        return

    df_results = pd.DataFrame(results)
    
    os.makedirs("results", exist_ok=True)
    output_path = "results/Window_Size_Analysis.xlsx"
    df_results.to_excel(output_path, index=False)
    print(f"\n✅ Analysis Complete! Data saved to {output_path}")

    # 5. Generate Comparison Plot
    plot_results(df_results)

def plot_results(df):
    """Generates a dual-axis chart: RMSE (Bar) and Accuracy (Line)"""
    print("📊 Generating Visualization...")
    os.makedirs("results/plots", exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot RMSE (Bars)
    bars = sns.barplot(data=df, x='Window_Days', y='RMSE', color='#4ECDC4', alpha=0.6, ax=ax1)
    ax1.set_ylabel('RMSE (Dollar Error)', color='#2c3e50', fontweight='bold')
    ax1.set_xlabel('Window Size (Days)', fontweight='bold')
    
    # Add values on bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height/2, 
                     f'${height:,.0f}', ha='center', va='center', color='white', fontweight='bold')

    # Plot Accuracy (Line) on secondary axis
    ax2 = ax1.twinx()
    # Accuracy = 100 - MAPE
    accuracy = 100 - df['MAPE']
    
    sns.lineplot(x=range(len(df)), y=accuracy, marker='o', markersize=10, color='#FF6B6B', linewidth=3, ax=ax2)
    ax2.set_ylabel('Model Accuracy (%)', color='#c0392b', fontweight='bold')
    
    # Dynamic Y-Limit for Accuracy to make the line look dramatic
    min_acc = min(accuracy)
    ax2.set_ylim(bottom=min_acc - 0.2, top=100.1)

    plt.title('Impact of Window Size (Days) on Performance', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plot_path = "results/plots/Window_Size_Comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Comparison Chart saved to {plot_path}")

if __name__ == "__main__":
    run_analysis()


