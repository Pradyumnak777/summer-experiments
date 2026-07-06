import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("checkpoints_new/channel_cond_v2/loss_log.csv")
df["smooth"] = df["loss"].rolling(200, min_periods=1).mean()   # 200-step moving avg

plt.figure(figsize=(10, 4))
plt.plot(df["global_step"], df["loss"], alpha=0.2, label="raw")
plt.plot(df["global_step"], df["smooth"], label="smoothed (200-step)")
plt.xlabel("step"); plt.ylabel("MSE loss"); plt.legend(); plt.tight_layout()
plt.savefig("checkpoints_new/channel_cond_v2/loss_curve.png", dpi=120)
print("saved loss_curve.png")