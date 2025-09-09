from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math

# --- VT annual % returns you posted (2009–2025 YTD) ---
returns = [13.09, 24.88, 53.81, -32.97, 26.63, 47.58, 37.96, -1.04, 31.52, 5.89, 8.43, 17.94, 34.99, 16.82, 2.70, 19.22, 53.54, -41.89, 18.67, 6.79, 1.49, 10.44, 49.12, -37.58, -32.65, -36.84, 101.95, 85.31, 20.63, 42.54, 42.54, 1.50, 10.58, 8.87, 64.99, -10.41, 26.17, 13.54]

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
