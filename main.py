import numpy as np
import matplotlib.pyplot as plt

#Leland model
S_max = 100 
T = 1.0
r = 0.05
sigma = 0.2
K = 50
lambda_ = 0.01  # Leland parameter (transaction costs)
M = 1000
N = 100
dt = T / M
dS = S_max / N  # stock price step

S = np.linspace(0, S_max, N+1)
V = np.maximum(S - K, 0)  # Payoff for a call option at maturity
V_old = V.copy()

def boundary_conditions(t):
    V[0] = 0
    V[-1] = S_max - K * np.exp(-r * t)

def capped_volatility(gamma_V, cap=1e3):
    gamma_capped = np.clip(gamma_V, -cap, cap)
    return sigma**2 * (1 + lambda_ * gamma_capped)

# finite difference method (explicit scheme)
for m in range(M):
    t = m * dt
    boundary_conditions(t)
    V_old = V.copy()

    for n in range(1, N):
        delta_V = (V_old[n+1] - V_old[n-1]) / (2 * dS)
        gamma_V = (V_old[n+1] - 2 * V_old[n] + V_old[n-1]) / (dS**2)
        vol = capped_volatility(gamma_V)
        #finite difference equation
        V[n] = V_old[n] + dt * (
            0.5 * vol * S[n]**2 * gamma_V - r * S[n] * delta_V + r * V_old[n]
        )

plt.plot(S, V, label='Option Value with Transaction Costs')
plt.xlabel('Stock Price')
plt.ylabel('Option Value')
plt.title('Nonlinear Black-Scholes PDE (Leland Model)')
plt.legend()
plt.show()
