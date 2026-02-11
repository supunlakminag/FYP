from preprocessing.data_loader import DataLoader
from rolling_window.roll_train import RollingWindowTrainer

# 1. Configuration
TICKER = "BTC-USD"
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"

# 2. Load Data
loader = DataLoader(TICKER, START_DATE, END_DATE)
df = loader.fetch_data()
df = loader.clean_data(df)

# 3. Run Rolling Window Training
# SET debug_mode=True for a quick test run!
trainer = RollingWindowTrainer(df, window_size=180, debug_mode=True)
predictions_df = trainer.run()

# 4. Save raw predictions
predictions_df.to_csv("results/predictions.csv", index=False)
print("✅ Predictions saved to results/predictions.csv")