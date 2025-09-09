from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math

# --- VT annual % returns you posted (2009–2025 YTD) ---
returns = [32.66, 13.08, -7.51, 17.12, 22.94, 3.68, -1.86, 8.50, 24.50,
           -9.76, 26.81, 16.59, 18.27, -18.00, 22.02, 16.49, 15.93]

arr = np.array(returns, dtype=float)
mu = arr.mean()
sigma = arr.std(ddof=1)
n = len(arr)

# --- Histogram + normal curve ("bell curve") ---
plt.figure(figsize=(9, 5.5))
plt.hist(arr, bins='auto', density=True, alpha=0.7)
x = np.linspace(arr.min() - 5, arr.max() + 5, 400)
pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
plt.plot(x, pdf, linewidth=2)
plt.title(f"VT Annual Returns Distribution (2009–2025 YTD)\n"
          f"n={n}, mean={mu:.2f}%, std={sigma:.2f}%")
plt.xlabel("Annual return (%)")
plt.ylabel("Density")
plt.tight_layout()

# --- Save locally (Downloads) ---
out_path = Path(r"C:\Users\Flopp\Downloads\vt_returns_bell_curve.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path)
print(f"Saved chart to: {out_path}")

plt.show()
