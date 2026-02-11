import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import numpy as np

# Set professional style
sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})

class ResultVisualizer:
    def __init__(self):
        self.metrics_file = "results/Final_Detailed_Comparison.xlsx"
        self.pred_file = "results/predictions.csv"
        self.sig_file = "results/signals.csv"
        os.makedirs("results/plots", exist_ok=True)

    def plot_accuracy_score(self):
        """
        Chart 1: The 'Report Card' (Accuracy %)
        Comparison of ALL models (B0, B1.1, B1.2, Evaluation)
        """
        print("📊 Generating Chart 1: Accuracy Score...")
        if not os.path.exists(self.metrics_file): return

        df = pd.read_excel(self.metrics_file)
        
        # Calculate Accuracy from MAPE
        df['MAPE_Val'] = df['MAPE (Error %)'].astype(str).str.replace('%','').astype(float)
        df['Accuracy'] = 100 - df['MAPE_Val']
        df['Accuracy'] = df['Accuracy'].apply(lambda x: max(x, 0))

        # Dynamic Data Loading
        models = df['Model_Type'].tolist()
        accuracy = df['Accuracy'].values
        
        # Define 4-Color Palette (Red -> Orange -> Yellow -> Green)
        # If you have fewer models, it just picks the first few colors
        palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71'] 
        colors = palette[:len(models)]

        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracy, color=colors, width=0.6)
        
        plt.title('Ablation Study: Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 115) 

        # Add Labels on top
        for bar in bars:
            height = bar.get_height()
            if height < 1: 
                label = "0%\n(Fail)"
            else:
                label = f"{height:.1f}%"
                
            plt.text(bar.get_x() + bar.get_width()/2, height + 2, 
                     label, ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

        plt.tight_layout()
        plt.savefig("results/plots/1_Accuracy_Comparison.png", dpi=300)
        plt.close()
        print("✅ Saved: results/plots/1_Accuracy_Comparison.png")

    def plot_error_cost(self):
        """
        Chart 2: The 'Cost of Being Wrong' (RMSE)
        Visualizes the drop in error across the 4 stages.
        """
        print("📊 Generating Chart 2: Dollar Error...")
        if not os.path.exists(self.metrics_file): return

        df = pd.read_excel(self.metrics_file)
        models = df['Model_Type'].tolist()
        rmse = df['RMSE (Error Cost)'].values
        
        plt.figure(figsize=(10, 6))
        
        # Palette: Red -> Green
        palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
        colors = palette[:len(models)]
        
        bars = plt.bar(models, rmse, color=colors, width=0.6)
        
        plt.title('Error Reduction Analysis (RMSE)', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Error Magnitude ($)', fontsize=12)
        
        # Add values on top
        for bar, val in zip(bars, rmse):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f"${val:,.0f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add annotation explaining the total improvement
        # Pointing from first bar to last bar
        if len(rmse) > 1:
            plt.annotate('Massive Improvement', 
                         xy=(len(rmse)-1, rmse[-1]), xytext=(0, rmse[0]*0.8),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=2))

        plt.tight_layout()
        plt.savefig("results/plots/2_Error_Comparison.png", dpi=300)
        plt.close()
        print("✅ Saved: results/plots/2_Error_Comparison.png")


    def plot_real_vs_predicted(self):
        print("📈 Generating Chart 3: Predictions...")
        if not os.path.exists(self.pred_file): return
        df = pd.read_csv(self.pred_file)
        df['Date'] = pd.to_datetime(df['Date'])
        subset = df.tail(100)
        plt.figure(figsize=(12, 6))
        plt.plot(subset['Date'], subset['Actual_Close'], label='Real Market Price', color='gray', linewidth=3, alpha=0.4)
        plt.plot(subset['Date'], subset['Predicted_Close'], label='Evaluation Model (Proposed)', color='#00b894', linewidth=2, linestyle='--')
        plt.title('Real-Time Prediction Capability (Last 100 Hours)', fontsize=16, fontweight='bold')
        plt.ylabel('Bitcoin Price (USD)')
        plt.xlabel('Time (Hour)')
        plt.legend(loc='upper left', fontsize=11, frameon=True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("results/plots/3_Prediction_Proof.png", dpi=300)
        plt.close()
        print("✅ Saved: results/plots/3_Prediction_Proof.png")

    def plot_signals_simplified(self):
        
        print("🚦 Generating Chart 4: Trading Signals...")
        if not os.path.exists(self.sig_file): return
        df = pd.read_csv(self.sig_file)
        df['Date'] = pd.to_datetime(df['Date'])
        subset = df.tail(80) 
        plt.figure(figsize=(12, 6))
        plt.plot(subset['Date'], subset['Actual_Close'], color='#2d3436', alpha=0.3, linewidth=2, label="Price")
        buys = subset[subset['Signal'] == 'BUY']
        plt.scatter(buys['Date'], buys['Actual_Close'], marker='^', color='#00b894', s=150, zorder=5, label='AI Says: BUY')
        sells = subset[subset['Signal'] == 'SELL']
        plt.scatter(sells['Date'], sells['Actual_Close'], marker='v', color='#d63031', s=150, zorder=5, label='AI Says: SELL')
        plt.title('Automated Trading Decisions (Buy vs Sell)', fontsize=16, fontweight='bold')
        plt.ylabel('Price (USD)')
        plt.legend(loc='best', frameon=True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:00'))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("results/plots/4_Trading_Decisions.png", dpi=300)
        plt.close()
        print("✅ Saved: results/plots/4_Trading_Decisions.png")

    def plot_cumulative_returns(self):
        print("💰 Generating Chart 5: Cumulative Returns...")
        if not os.path.exists(self.sig_file): return
        df = pd.read_csv(self.sig_file)
        df['Pct_Change'] = df['Actual_Close'].pct_change().shift(-1).fillna(0)
        df['Strategy_Return'] = 0.0
        df.loc[df['Signal'] == 'BUY', 'Strategy_Return'] = df['Pct_Change']
        df.loc[df['Signal'] == 'SELL', 'Strategy_Return'] = -df['Pct_Change']
        df['Buy_Hold_Curve'] = (1 + df['Pct_Change']).cumprod() * 1000
        df['Strategy_Curve'] = (1 + df['Strategy_Return']).cumprod() * 1000
        plt.figure(figsize=(12, 6))
        plt.plot(df['Buy_Hold_Curve'], label='Buy & Hold Bitcoin', color='gray', alpha=0.5, linestyle='--')
        plt.plot(df['Strategy_Curve'], label='Model', color='#00b894', linewidth=2)
        plt.title('Profitability Analysis: $1,000 Investment', fontsize=16, fontweight='bold')
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.xlabel('Time Steps (Hours)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plots/5_Cumulative_Returns.png", dpi=300)
        plt.close()
        print("✅ Saved: results/plots/5_Cumulative_Returns.png")

    def plot_error_histogram(self):
        print("🔔 Generating Chart 6: Error Histogram...")
        if not os.path.exists(self.pred_file): return
        df = pd.read_csv(self.pred_file)
        residuals = df['Actual_Close'] - df['Predicted_Close']
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=50, kde=True, color='#6c5ce7')
        plt.title('Error Distribution (Residual Analysis)', fontsize=16, fontweight='bold')
        plt.xlabel('Prediction Error ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.text(0, plt.gca().get_ylim()[1]*0.9, " Zero Error Line", color='red', fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/plots/6_Error_Distribution.png", dpi=300)
        plt.close()
        print("✅ Saved: results/plots/6_Error_Distribution.png")

    def plot_signal_counts(self):
        print("📊 Generating Chart 7: Signal Counts...")
        if not os.path.exists(self.sig_file): return
        df = pd.read_csv(self.sig_file)
        counts = df['Signal'].value_counts()
        plt.figure(figsize=(8, 6))
        colors = {'HOLD': '#b2bec3', 'BUY': '#00b894', 'SELL': '#d63031'}
        order = [x for x in ['HOLD', 'BUY', 'SELL'] if x in counts.index]
        plt.bar(order, [counts[x] for x in order], color=[colors[x] for x in order])
        plt.title('Trading Frequency (Operational Check)', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Hours', fontsize=12)
        for i, count in enumerate([counts[x] for x in order]):
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/plots/7_Signal_Counts.png", dpi=300)
        plt.close()
        print("✅ Saved: results/plots/7_Signal_Counts.png")

if __name__ == "__main__":
    viz = ResultVisualizer()
    viz.plot_accuracy_score()
    viz.plot_error_cost()
    viz.plot_real_vs_predicted()
    viz.plot_signals_simplified()
    viz.plot_cumulative_returns()
    viz.plot_error_histogram()
    viz.plot_signal_counts()