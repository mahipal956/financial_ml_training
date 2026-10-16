# Strategy Creation & ML Training Guide

Authoritative operational blueprint for creating, training, validating, and deploying institutional-grade financial machine learning strategies within the `financial_ml_training` pipeline (Marcos López de Prado - *Advances in Financial Machine Learning* paradigm).

---

## 1. Architecture Overview: Two-Stage Meta-Labeling

```
                     ┌────────────────────────────────┐
                     │         Market Data Feed       │
                     │    (Tick / Volume / Time Bars) │
                     └───────────────┬────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │ Symmetric CUSUM Filter│
                         └───────────┬───────────┘
                                     │ (Triggered Events)
                         ┌───────────┴───────────┐
                         │ Triple Barrier Method │
                         └───────────┬───────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         Primary Model         │               │          Meta-Model           │
│     (Directional Recall)      │               │   (Precision Veto & Bet Size) │
├───────────────────────────────┤               ├───────────────────────────────┤
│ • Decides LONG (+1) or SHORT  │               │ • Predicts probability of     │
│ • Maximizes trade opportunities│──────────────>│   primary model being right   │
│ • High recall, low false neg  │               │ • Vetoes false breakouts      │
└───────────────────────────────┘               │ • Sizes position [0.0 to 1.0] │
                                                └───────────────┬───────────────┘
                                                                │
                                                ┌───────────────┴───────────────┐
                                                │      ONNX & Config Export     │
                                                │     (cTrader C# Execution)    │
                                                └───────────────────────────────┘
```

The system strictly decouples **direction** from **position sizing**:
1. **Primary Model:** Decides *when* and *which direction* to enter.
2. **Meta-Model (Secondary):** Acts as a risk supervisor that predicts whether the primary model's trade will succeed (`y=1`) or fail (`y=0`), filtering out noise and sizing the position.

---

## 2. Strategy Creation Workflow

### Step 2.1: Event-Driven Sampling (Symmetric CUSUM Filter)
Instead of sampling at arbitrary clock times (which generates heteroskedasticity and non-stationarity), sample events when cumulative price variations exceed a threshold:
$$\Delta p_t = p_t - p_{t-1}$$
$$S_t^+ = \max(0, S_{t-1}^+ + \Delta p_t), \quad S_t^- = \min(0, S_{t-1}^- + \Delta p_t)$$
An event is triggered when $S_t^+ \ge h$ or $S_t^- \le -h$, where $h$ is dynamically pegged to daily volatility $\sigma_t$.

### Step 2.2: Triple Barrier Labeling
Each sampled event $t_0$ is evaluated against three barriers:
* **Upper Barrier:** Profit-taking horizontal limit: $p_{t_0} \times (1 + pt \cdot \sigma_{t_0})$
* **Lower Barrier:** Stop-loss horizontal limit: $p_{t_0} \times (1 - sl \cdot \sigma_{t_0})$
* **Vertical Barrier ($t_1$):** Maximum holding horizon (bars / time elapsed).

### Step 2.3: Stationary Feature Engineering
Preserve historical long-term memory while guaranteeing stationarity:
* **Fractional Differentiation:** Find the minimum degree $d \in [0, 1]$ that passes the Augmented Dickey-Fuller (ADF) test ($p < 0.05$) while preserving maximum correlation with the raw series.
* **Microstructural Features:** Bid-ask spreads, order flow imbalance, Roll measure, Kyle lambda.
* **Volatility Features:** Garman-Klass, Parkinson, and exponential standard deviation measures.

---

## 3. Training & Validation Protocol

### Step 3.1: Sample Weights & Concurrency
Financial events overlap in time, creating non-independent observations:
* **Concurrency ($c_t$):** Number of active concurrent barriers at timestamp $t$.
* **Uniqueness ($u_{i, t}$):** Inverse concurrency: $u_{i, t} = \frac{1}{c_t}$.
* **Sample Weight ($w_i$):** Product of average uniqueness and absolute return attributed to the event, with optional exponential time-decay weighting.

### Step 3.2: Purged K-Fold Cross-Validation
Standard k-fold cross-validation suffers from massive informational leakage. We enforce:
* **Purging:** Removing training labels whose event horizon $[t_0, t_1]$ overlaps with the test fold.
* **Embargo:** Adding a temporal embargo buffer directly following the test set to eliminate autoregressive memory leakage.

### Step 3.3: Feature Importance Stack
Run the 4-layer importance verification before accepting features:
1. **MDI (Mean Decrease Impurity):** In-sample tree split contributions.
2. **MDA (Mean Decrease Accuracy):** Out-of-sample permutation scoring on purged folds.
3. **SFI (Single Feature Importance):** Cross-entropy drop per isolated feature (prevents substitution effects).
4. **PCA Orthogonalization:** Eigenvector decomposition to remove multicollinearity.

---

## 4. Bet Sizing & Real-Time C# Deployment

### Bet Sizing Formula (López de Prado)
Transform predicted meta-probability $p = P(y=1)$ into a continuous position size $s \in [-1, 1]$:
$$z = \frac{p - \frac{1}{2}}{\sqrt{p(1 - p)}}$$
$$s = 2 \cdot \Phi(z) - 1$$
Where $\Phi$ is the standard normal cumulative distribution function.

### Real-Time cTrader C# Clamping
On every tick/bar in cAlgo, the normalized trade volume is bounded:
```csharp
double rawVolume = (Account.Equity * RiskPercentage / 100.0) / (stopLossPips * Symbol.PipValue);
double normalizedVolume = Symbol.NormalizeVolumeInUnits(rawVolume * Math.Abs(betSize), RoundingMode.Down);
if (normalizedVolume < Symbol.VolumeInUnitsMin) normalizedVolume = Symbol.VolumeInUnitsMin;
if (normalizedVolume > Symbol.VolumeInUnitsMax) normalizedVolume = Symbol.VolumeInUnitsMax;
```

---

## 5. Pre-Flight Verification Checklist

Before deploying any trained model to live cTrader accounts:
- [ ] **ADF Test:** All features pass Augmented Dickey-Fuller stationarity test ($p < 0.05$).
- [ ] **No Future Leakage:** Target labels strictly computed on $[t_0, t_1]$ forward bars; features strictly computed on $\le t_0$.
- [ ] **Purged CV:** Out-of-sample Sharpe ratio evaluated with PurgedKFold and post-test embargo.
- [ ] **ONNX Numerical Equivalence:** Python output vs. ONNX Runtime C# inferences have maximum absolute difference $< 10^{-5}$.
- [ ] **Safe Volume Clamping:** Volume normalized and clamped to `VolumeInUnitsMin` and `VolumeInUnitsMax`.
