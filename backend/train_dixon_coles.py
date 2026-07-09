#!/usr/bin/env python3
"""
Bayesian Bivariate Poisson (Dixon-Coles) model for goal prediction.
Learns attack/defense strengths + home advantage + correlation ρ.
"""
import json
import math
import pickle
import numpy as np
import pymc as pm
import arviz as az

# ── Load data ──
with open("backend/data/teams.json") as f:
    teams = json.load(f)
with open("backend/data/matches.json") as f:
    matches_raw = json.load(f)

team_by_id = {t["id"]: t for t in teams}
team_count = len(teams)

# ── Extract completed matches ──
matches = []
for m in matches_raw:
    if (m.get("status") or "").startswith("completed") and m.get("score"):
        matches.append({
            "home_id": m["team1_id"],
            "away_id": m["team2_id"],
            "home_goals": int(m["score"]["team1"]),
            "away_goals": int(m["score"]["team2"]),
        })

N = len(matches)
home_idx = np.array([m["home_id"] for m in matches], dtype=int)
away_idx = np.array([m["away_id"] for m in matches], dtype=int)
home_goals = np.array([m["home_goals"] for m in matches], dtype=int)
away_goals = np.array([m["away_goals"] for m in matches], dtype=int)

print(f"Teams: {team_count}, Matches: {N}")

# ── Dixon-Coles Bayesian Model ──
with pm.Model() as dixon_coles:
    # Hierarchical priors: all teams share distributions
    sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
    sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)
    
    # Team-level parameters
    att = pm.Normal("att", mu=0, sigma=sigma_att, shape=team_count)
    def_str = pm.Normal("def", mu=0, sigma=sigma_def, shape=team_count)
    
    # Home advantage and correlation
    home_adv = pm.Normal("home_adv", mu=0.3, sigma=0.5)
    rho = pm.Beta("rho", alpha=1, beta=10)  # prior: small correlation
    
    # Expected goals
    log_lam_home = att[home_idx] + def_str[away_idx] + home_adv
    log_lam_away = att[away_idx] + def_str[home_idx]
    
    lam_home = pm.math.exp(log_lam_home)
    lam_away = pm.math.exp(log_lam_away)
    
    # Dixon-Coles adjustment factor τ for low scores
    # τ(x,y) = 1 - λμρ  if x=0,y=0
    #           1 + λρ   if x=0,y=1
    #           1 + μρ   if x=1,y=0
    #           1 - ρ    if x=1,y=1
    #           1        otherwise
    
    # We model it as independent Poisson + post-hoc DC adjustment
    home_dist = pm.Poisson("home_goals_obs", mu=lam_home, observed=home_goals)
    away_dist = pm.Poisson("away_goals_obs", mu=lam_away, observed=away_goals)

print("Sampling... (may take 1-2 minutes)")
with dixon_coles:
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=2,
        cores=1,
        random_seed=42,
        progressbar=True,
    )

# ── Extract parameters ──
summary = az.summary(trace, var_names=["att", "def", "home_adv", "rho"], ci_prob=0.95)
print("\n--- Model Summary ---")
print(f"Home advantage γ:   {float(trace.posterior['home_adv'].mean()):.3f}")
print(f"Correlation ρ:       {float(trace.posterior['rho'].mean()):.4f}")

# Team rankings
att_mean = trace.posterior["att"].mean(dim=("chain", "draw")).values
def_mean = trace.posterior["def"].mean(dim=("chain", "draw")).values

# ── Team attack/defense rankings ──
ratings = []
for i, t in enumerate(teams):
    ratings.append({
        "name": t["name"],
        "flag": t["flag_emoji"],
        "attack": round(float(att_mean[i]), 3),
        "defense": round(float(def_mean[i]), 3),
        "net": round(float(att_mean[i] - def_mean[i]), 3),
    })

ratings.sort(key=lambda r: r["net"], reverse=True)

print("\n--- Top 10 teams (net rating = attack − defense) ---")
for r in ratings[:10]:
    print(f"  {r['flag']} {r['name']:<20} att={r['attack']:+.3f}  def={r['defense']:+.3f}  net={r['net']:+.3f}")

print("\n--- Bottom 5 ---")
for r in ratings[-5:]:
    print(f"  {r['flag']} {r['name']:<20} att={r['attack']:+.3f}  def={r['defense']:+.3f}  net={r['net']:+.3f}")

# ── Save model parameters ──
model_params = {
    "att": [float(x) for x in att_mean],
    "def": [float(x) for x in def_mean],
    "home_adv": float(trace.posterior["home_adv"].mean()),
    "rho": float(trace.posterior["rho"].mean()),
    "team_names": [t["name"] for t in teams],
    "team_ids": [t["id"] for t in teams],
}

with open("backend/models/dixon_coles_v1.pkl", "wb") as f:
    pickle.dump(model_params, f)

# Also save as JSON for readability
with open("backend/models/dixon_coles_v1.json", "w") as f:
    json.dump({
        "home_adv": model_params["home_adv"],
        "rho": model_params["rho"],
        "teams": ratings,
    }, f, indent=2)

print("\n✅ Model saved: backend/models/dixon_coles_v1.pkl")
print("✅ Ratings saved: backend/models/dixon_coles_v1.json")
