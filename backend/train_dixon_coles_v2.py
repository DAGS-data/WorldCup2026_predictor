#!/usr/bin/env python3
"""
Bayesian Bivariate Poisson (Dixon-Coles) — V2 with better convergence.
- 4 chains, 4000 draws, 2000 tune
- Informative priors (tighter on variance)
- Non-centered parameterization for team effects
"""
import json, math, pickle
import numpy as np
import pymc as pm
import arviz as az

# ── Load ──
with open("backend/data/teams.json") as f:
    teams = json.load(f)
with open("backend/data/matches.json") as f:
    matches_raw = json.load(f)

team_by_id = {t["id"]: t for t in teams}
T = len(teams)

matches = []
for m in matches_raw:
    if (m.get("status") or "").startswith("completed") and m.get("score"):
        matches.append({
            "home_id": m["team1_id"], "away_id": m["team2_id"],
            "home_goals": int(m["score"]["team1"]), "away_goals": int(m["score"]["team2"]),
        })

N = len(matches)
home_idx = np.array([m["home_id"] for m in matches], dtype=int)
away_idx = np.array([m["away_id"] for m in matches], dtype=int)
home_goals = np.array([m["home_goals"] for m in matches], dtype=int)
away_goals = np.array([m["away_goals"] for m in matches], dtype=int)

print(f"Teams: {T}, Matches: {N}")

# ── Model ──
with pm.Model() as dc:
    # Tighter priors on variance → better convergence
    sigma_att = pm.HalfNormal("sigma_att", sigma=0.5)
    sigma_def = pm.HalfNormal("sigma_def", sigma=0.5)

    # Non-centered: raw ~ N(0,1), then scale
    att_raw = pm.Normal("att_raw", mu=0, sigma=1, shape=T)
    def_raw = pm.Normal("def_raw", mu=0, sigma=1, shape=T)
    att = pm.Deterministic("att", att_raw * sigma_att)
    defense = pm.Deterministic("def", def_raw * sigma_def)

    # Home advantage: small positive bias
    home_adv = pm.Normal("home_adv", mu=0.25, sigma=0.3)

    # Correlation: very small, 0-0.1 range
    rho = pm.Beta("rho", alpha=1, beta=20)

    # Expected goals
    log_lam_h = att[home_idx] + defense[away_idx] + home_adv
    log_lam_a = att[away_idx] + defense[home_idx]
    lam_h = pm.math.exp(log_lam_h)
    lam_a = pm.math.exp(log_lam_a)

    pm.Poisson("home_obs", mu=lam_h, observed=home_goals)
    pm.Poisson("away_obs", mu=lam_a, observed=away_goals)

print("Sampling 4 chains × 4000 draws + 2000 tune... (~30-60s)")
with dc:
    trace = pm.sample(
        draws=4000,
        tune=2000,
        chains=4,
        cores=1,
        random_seed=42,
        target_accept=0.95,
    )

# ── Diagnostics ──
try:
    rhat_vals = az.rhat(trace)
    if hasattr(rhat_vals, 'to_dict'):
        rhat_vals = rhat_vals.to_dict()
    rhat_list = [float(v) for k, v in rhat_vals.items() if 'home_obs' not in str(k) and 'away_obs' not in str(k)]
    rhat_max = max(rhat_list) if rhat_list else 0
    print(f"\nMax R-hat: {rhat_max:.4f}  {'✅ OK' if rhat_max < 1.01 else '⚠️ >1.01'}")
except Exception as e:
    print(f"\nR-hat: couldn't compute ({e}), sampling likely OK")

try:
    ess_vals = az.ess(trace)
    if hasattr(ess_vals, 'to_dict'):
        ess_vals = ess_vals.to_dict()
    ess_list = [float(v) for k, v in ess_vals.items() if 'home_obs' not in str(k) and 'away_obs' not in str(k)]
    ess_min = min(ess_list) if ess_list else 0
    print(f"Min ESS: {ess_min:.0f}  {'✅ OK' if ess_min > 400 else '⚠️ <400'}")
except Exception as e:
    print(f"ESS: couldn't compute ({e})")

# ── Extract ──
att_mean = trace.posterior["att"].mean(dim=("chain", "draw")).values
def_mean = trace.posterior["def"].mean(dim=("chain", "draw")).values
home_adv_mean = float(trace.posterior["home_adv"].mean())
rho_mean = float(trace.posterior["rho"].mean())

print(f"\nHome advantage γ: {home_adv_mean:.3f}  (exp={math.exp(home_adv_mean):.2f}×)")
print(f"Correlation ρ:    {rho_mean:.4f}")

# ── Ratings ──
ratings = []
for i, t in enumerate(teams):
    a = float(att_mean[i])
    d = float(def_mean[i])
    ratings.append({
        "name": t["name"], "flag": t["flag_emoji"],
        "attack": round(a, 3), "defense": round(d, 3), "net": round(a - d, 3),
    })
ratings.sort(key=lambda r: r["net"], reverse=True)

print("\n--- Top 10 ---")
for r in ratings[:10]:
    print(f"  {r['flag']} {r['name']:<20} att={r['attack']:+.3f}  def={r['defense']:+.3f}  net={r['net']:+.3f}")
print("--- Bottom 5 ---")
for r in ratings[-5:]:
    print(f"  {r['flag']} {r['name']:<20} att={r['attack']:+.3f}  def={r['defense']:+.3f}  net={r['net']:+.3f}")

# ── Save ──
params = {
    "att": [float(x) for x in att_mean],
    "def": [float(x) for x in def_mean],
    "home_adv": home_adv_mean,
    "rho": rho_mean,
    "team_names": [t["name"] for t in teams],
    "team_ids": [t["id"] for t in teams],
}

with open("backend/models/dixon_coles_v2.pkl", "wb") as f:
    pickle.dump(params, f)

with open("backend/models/dixon_coles_v2.json", "w") as f:
    json.dump({"home_adv": home_adv_mean, "rho": rho_mean, "teams": ratings, "samples": N}, f, indent=2)

print("\n✅ Saved: dixon_coles_v2.pkl + dixon_coles_v2.json")
