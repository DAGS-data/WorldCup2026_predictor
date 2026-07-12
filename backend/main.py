"""World Cup 2026 Predictor — FastAPI backend (pre-computed data for instant response)."""

import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from performance import calculate_performance_rating
from predictor import calculate_match_probabilities, get_prediction_factors, simulate_tournament
from feature_engineering import build_team_history, compute_match_features
from models.xgboost_predictor import KnockoutPredictor
from models.logistic_predictor import LogisticPredictor

# --- Environment ---
IS_PRODUCTION = os.environ.get("ENV", "").lower() in ("production", "prod")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://worldcup2026-predictor.seenode.app").split(",")

# --- Security headers middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

# Init predictors (lazy load)
_xgb_predictor = None
_logistic_predictor = None

def get_xgb_predictor():
    global _xgb_predictor
    if _xgb_predictor is None:
        _xgb_predictor = KnockoutPredictor()
    return _xgb_predictor

def get_logistic_predictor():
    global _logistic_predictor
    if _logistic_predictor is None:
        _logistic_predictor = LogisticPredictor()
    return _logistic_predictor

DATA_DIR = Path(__file__).parent / "data"

app = FastAPI(
    title="WC 2026 Predictor API",
    version="2.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# Security headers (applied to every response)
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Fast pre-computed data loading (zero computation on each request)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_teams_enriched() -> list[dict]:
    """Load pre-computed teams with performance ratings. Instant."""
    path = DATA_DIR / "teams_enriched.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Fallback
    with open(DATA_DIR / "teams.json") as f:
        teams = json.load(f)
    return [enrich_team(t) for t in teams]


@lru_cache(maxsize=1)
def load_matches_enriched() -> list[dict]:
    """Load pre-computed matches with predictions. Instant."""
    path = DATA_DIR / "matches_enriched.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Fallback
    from predictor import calculate_match_probabilities, get_prediction_factors
    with open(DATA_DIR / "teams.json") as f:
        teams = {t["id"]: t for t in json.load(f)}
    with open(DATA_DIR / "matches.json") as f:
        matches = json.load(f)
    for m in matches:
        t1 = teams.get(m["team1_id"])
        t2 = teams.get(m["team2_id"])
        if t1 and t2:
            probs = calculate_match_probabilities(
                t1["elo_rating"], t2["elo_rating"],
                t1.get("is_host", False), t2.get("is_host", False))
            factors = get_prediction_factors(t1, t2)
            m["team1_name"] = t1["name"]
            m["team1_flag"] = t1["flag_emoji"]
            m["team2_name"] = t2["name"]
            m["team2_flag"] = t2["flag_emoji"]
            m["prediction"] = {**probs, "key_factors": factors}
    return matches


@lru_cache(maxsize=1)
def load_teams_raw() -> list[dict]:
    with open(DATA_DIR / "teams.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_matches_raw() -> list[dict]:
    with open(DATA_DIR / "matches.json") as f:
        return json.load(f)


def enrich_team(team: dict) -> dict:
    perf = calculate_performance_rating(team["elo_rating"], team.get("recent_form", []))
    return {**team, "performance": perf}


# ---------------------------------------------------------------------------
# Endpoints — serving pre-computed data
# ---------------------------------------------------------------------------

@app.get("/api/teams")
def get_teams():
    return load_teams_enriched()


@app.get("/api/teams/{team_id}")
def get_team(team_id: int):
    teams = load_teams_enriched()
    team = next((t for t in teams if t["id"] == team_id), None)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@app.get("/api/matches")
def get_matches():
    return load_matches_enriched()


@app.get("/api/matches/{match_id}")
def get_match(match_id: int):
    matches = load_matches_enriched()
    match = next((m for m in matches if m["id"] == match_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@app.get("/api/groups")
def get_groups():
    teams = load_teams_enriched()
    matches = load_matches_enriched()
    teams_lookup = {t["id"]: t for t in teams}
    groups: dict[str, dict] = {}

    for team in teams:
        g = team["group"]
        if g not in groups:
            groups[g] = {"group": g, "teams": [], "matches": []}
        groups[g]["teams"].append(team)

    for match in matches:
        g = match.get("group", "")
        if g in groups:
            groups[g]["matches"].append(match)

    return sorted(groups.values(), key=lambda x: x["group"])


@app.get("/api/bracket")
def get_bracket():
    teams = load_teams_raw()
    matches = load_matches_raw()
    teams_by_group: dict[str, list[dict]] = {}
    for team in teams:
        g = team["group"]
        teams_by_group.setdefault(g, []).append(team)
    return simulate_tournament(teams_by_group, matches)


@app.get("/api/predict")
def predict_match(team1_id: int, team2_id: int):
    teams = load_teams_enriched()
    t1 = next((t for t in teams if t["id"] == team1_id), None)
    t2 = next((t for t in teams if t["id"] == team2_id), None)
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="Team not found")

    probs = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False))
    factors = get_prediction_factors(t1, t2)

    return {
        "team1": t1,
        "team2": t2,
        "prediction": {**probs, "key_factors": factors},
    }


@app.get("/api/retrain")
def retrain():
    load_teams_enriched.cache_clear()
    load_matches_enriched.cache_clear()
    load_teams_raw.cache_clear()
    load_matches_raw.cache_clear()
    return {"status": "ok", "message": "Caches cleared"}


@app.get("/api/predict/v2")
def predict_match_v2(team1_id: int, team2_id: int):
    """XGBoost-powered prediction: P(team advances) with 38 features + SHAP."""
    teams = load_teams_enriched()
    t1 = next((t for t in teams if t["id"] == team1_id), None)
    t2 = next((t for t in teams if t["id"] == team2_id), None)
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="Team not found")

    matches = load_matches_raw()
    teams_by_id = {t["id"]: t for t in teams}
    team_history = build_team_history(matches, teams_by_id)
    t1_hist = team_history.get(team1_id, [])
    t2_hist = team_history.get(team2_id, [])

    # --- Dual-prediction symmetrization ---
    # Predict P(team1 advances) from BOTH perspectives and average.
    # Reason: XGBoost features are directional (elo_diff, momentum_diff, etc.)
    # and tree models don't guarantee complementary probabilities. Dual
    # prediction + averaging eliminates order bias — swapping teams yields
    # exactly complementary probabilities.
    features_fwd = compute_match_features(
        team1_id, team2_id, t1, t2,
        t1_hist, t2_hist,
        "round_of_16", "2026-07-04"
    )
    features_rev = compute_match_features(
        team2_id, team1_id, t2, t1,
        t2_hist, t1_hist,
        "round_of_16", "2026-07-04"
    )

    try:
        xgb = get_xgb_predictor()
        prob_fwd = xgb.predict(features_fwd)       # P(team1 advances) from team1's view
        prob_rev = xgb.predict(features_rev)       # P(team2 advances) from team2's view
        # Symmetrize: average forward prediction with (1 - reverse prediction)
        prob = round((prob_fwd + (1.0 - prob_rev)) / 2.0, 4)
        explanation = xgb.explain(features_fwd)    # SHAP from team1's perspective
    except Exception as e:
        prob = None
        explanation = {"error": str(e)}

    poisson = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False)
    )

    return {
        "team1": {"name": t1["name"], "flag": t1["flag_emoji"], "elo": t1["elo_rating"], "fifa_rank": t1.get("fifa_rank")},
        "team2": {"name": t2["name"], "flag": t2["flag_emoji"], "elo": t2["elo_rating"], "fifa_rank": t2.get("fifa_rank")},
        "v2_prediction": {
            "team1_advance_prob": prob,
            "model": "XGBoost (38 features)",
            "features_used": len(features_fwd),
        },
        "v1_baseline": {
            "team1_win_prob": poisson["team1_win_prob"],
            "draw_prob": poisson["draw_prob"],
            "team2_win_prob": poisson["team2_win_prob"],
            "model": "ELO + Poisson",
        },
        "explanation": explanation,
    }


@app.get("/api/predict/v3")
def predict_match_v3(team1_id: int, team2_id: int):
    """Logistic Regression prediction: P(team advances) + feature contributions.
    
    Replaces XGBoost with L2-regularized logistic regression.
    No overfitting, naturally calibrated probabilities, interpretable.
    """
    teams = load_teams_enriched()
    t1 = next((t for t in teams if t["id"] == team1_id), None)
    t2 = next((t for t in teams if t["id"] == team2_id), None)
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="Team not found")

    matches = load_matches_raw()
    teams_by_id = {t["id"]: t for t in teams}
    team_history = build_team_history(matches, teams_by_id)
    t1_hist = team_history.get(team1_id, [])
    t2_hist = team_history.get(team2_id, [])

    features_fwd = compute_match_features(
        team1_id, team2_id, t1, t2, t1_hist, t2_hist, "round_of_16", "2026-07-04")
    features_rev = compute_match_features(
        team2_id, team1_id, t2, t1, t2_hist, t1_hist, "round_of_16", "2026-07-04")

    try:
        lr = get_logistic_predictor()
        prob_fwd = lr.predict(features_fwd)
        prob_rev = lr.predict(features_rev)
        prob = round((prob_fwd + (1.0 - prob_rev)) / 2.0, 4)
        explanation = lr.explain(features_fwd)
    except Exception as e:
        prob = None
        explanation = {"error": str(e)}

    poisson = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False))

    return {
        "team1": {"name": t1["name"], "flag": t1["flag_emoji"], "elo": t1["elo_rating"], "fifa_rank": t1.get("fifa_rank")},
        "team2": {"name": t2["name"], "flag": t2["flag_emoji"], "elo": t2["elo_rating"], "fifa_rank": t2.get("fifa_rank")},
        "v3_prediction": {
            "team1_advance_prob": prob,
            "model": "Logistic Regression (L2, C=0.1, 38 features)",
            "features_used": len(features_fwd),
        },
        "v1_baseline": {
            "team1_win_prob": poisson["team1_win_prob"],
            "draw_prob": poisson["draw_prob"],
            "team2_win_prob": poisson["team2_win_prob"],
            "model": "ELO + Poisson",
        },
        "explanation": explanation,
    }


@app.get("/api/model-info")
def model_info():
    try:
        xgb = get_xgb_predictor()
        metrics = xgb.get_metrics()
        return {
            "model": "XGBoost", "type": "Binary classifier — P(advance)",
            "features": metrics.get("features", 0),
            "accuracy": metrics.get("accuracy"),
            "roc_auc": metrics.get("auc"),
            "brier_score": metrics.get("brier"),
            "val_samples": metrics.get("val_samples"),
            "status": "loaded" if metrics.get("loaded") else "not loaded",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/bracket/v2")
def bracket_v2():
    """Simulate the full knockout bracket using XGBoost dual-prediction.

    Starts from R16 enriched matches, simulates QF → SF → Final
    by calling the XGBoost predictor for each matchup. Returns the
    complete bracket tree with winners, probabilities, and flags.
    """
    teams_raw = load_teams_raw()
    teams_lookup = {t["id"]: t for t in teams_raw}
    matches_enriched = load_matches_enriched()
    matches_raw = load_matches_raw()

    team_history = build_team_history(matches_raw, teams_lookup)

    xgb = get_xgb_predictor()

    def predict_match(team_a, team_b, stage, date):
        """Return (prob_a_advances, prob_b_advances, winner_id)."""
        aid = team_a["id"]; bid = team_b["id"]
        h_a = team_history.get(aid, []); h_b = team_history.get(bid, [])
        try:
            fwd = compute_match_features(aid, bid, team_a, team_b, h_a, h_b, stage, date)
            rev = compute_match_features(bid, aid, team_b, team_a, h_b, h_a, stage, date)
            pf = xgb.predict(fwd); pr = xgb.predict(rev)
            pa = round((pf + (1.0 - pr)) / 2.0, 4)
            pb = round(1.0 - pa, 4)
            return pa, pb, (aid if pa >= 0.5 else bid)
        except Exception as e:
            return 0.5, 0.5, aid  # fallback

    # Get R16 matches
    r16_matches = [m for m in matches_enriched if m.get("stage") == "round_of_16"]
    # Sort by id for consistent ordering
    r16_matches.sort(key=lambda m: m.get("id", 0))

    rounds = {"round_of_16": [], "quarter_finals": [], "semi_finals": [], "final": []}

    # Helper: determine real winner from completed match
    def get_match_result(raw):
        """Return (winner_id, score_str, status_str) from a completed match, or None."""
        if not raw or not raw.get("score"):
            return None
        s1 = int(raw["score"]["team1"]); s2 = int(raw["score"]["team2"])
        status = raw.get("status", "")
        t1_won = s1 > s2
        if status.endswith("_penalties"):
            pen = raw["score"].get("penalties", "")
            if pen:
                p1, p2 = map(int, pen.split("-"))
                t1_won = p1 > p2
                return (raw["team1_id"] if t1_won else raw["team2_id"],
                        f"{s1}-{s2} ({p1}-{p2} pen)", status)
        if status.endswith("_aet"):
            return (raw["team1_id"] if t1_won else raw["team2_id"],
                    f"{s1}-{s2} (aet)", status)
        return (raw["team1_id"] if t1_won else raw["team2_id"],
                f"{s1}-{s2}", status)

    # --- R16: use real results if completed, else prediction ---
    r16_winners = []
    for m in r16_matches:
        t1id = m.get("team1_id"); t2id = m.get("team2_id")
        t1 = teams_lookup.get(t1id, {}); t2 = teams_lookup.get(t2id, {})
        raw = next((rm for rm in matches_raw if rm.get("id") == m.get("id")), None)
        is_completed = raw and raw.get("status", "").startswith("completed")
        
        if is_completed:
            result = get_match_result(raw)
            if result:
                winner_id, score_display, match_status = result
                rounds["round_of_16"].append({
                    "match_id": m.get("id"),
                    "team1": team_info(t1),
                    "team2": team_info(t2),
                    "team1_prob": 100 if winner_id == t1id else 0,
                    "team2_prob": 0 if winner_id == t1id else 100,
                    "winner_id": winner_id,
                    "score": score_display,
                    "status": match_status,
                })
                r16_winners.append(teams_lookup.get(winner_id, t1))
                continue
        
        # Fallback to prediction
        xgb_pred = m.get("xgb_prediction", {})
        p1 = xgb_pred.get("team1_advance_prob", 50)
        p2 = xgb_pred.get("team2_advance_prob", 50)
        winner_id = t1id if p1 >= p2 else t2id
        rounds["round_of_16"].append({
            "match_id": m.get("id"),
            "team1": team_info(t1),
            "team2": team_info(t2),
            "team1_prob": p1,
            "team2_prob": p2,
            "winner_id": winner_id,
            "status": "scheduled",
        })
        r16_winners.append(teams_lookup.get(winner_id, teams_lookup.get(t1id)))

    # --- QF: use actual matchups from matches_enriched (IDs 96-99) ---
    qf_matches = sorted(
        [m for m in matches_enriched if m.get("stage") == "quarter_finals" and m.get("team1_id")],
        key=lambda m: m.get("id", 0)
    )
    qf_winners = []
    if qf_matches:
        for m in qf_matches:
            t1id = m.get("team1_id"); t2id = m.get("team2_id")
            t1 = teams_lookup.get(t1id, {}); t2 = teams_lookup.get(t2id, {})
            if not t1 or not t2:
                continue
            pa, pb, wid = predict_match(t1, t2, "quarter_finals", m.get("date", "2026-07-10"))
            rounds["quarter_finals"].append({
                "match_id": m.get("id"),
                "team1": team_info(t1),
                "team2": team_info(t2),
                "team1_prob": round(pa*100, 1),
                "team2_prob": round(pb*100, 1),
                "winner_id": wid,
                "status": "scheduled",
            })
            qf_winners.append(t1 if wid == t1["id"] else t2)
    else:
        # Fallback: chain from R16 winners
        for i in range(0, 8, 2):
            ta = r16_winners[i]; tb = r16_winners[i+1]
            pa, pb, wid = predict_match(ta, tb, "quarter_finals", "2026-07-10")
            rounds["quarter_finals"].append({
                "team1": team_info(ta),
                "team2": team_info(tb),
                "team1_prob": round(pa*100, 1),
                "team2_prob": round(pb*100, 1),
                "winner_id": wid,
                "status": "scheduled",
            })
            qf_winners.append(ta if wid == ta["id"] else tb)

    # --- SF ---
    sf_winners = []
    for i in range(0, 4, 2):
        ta = qf_winners[i]; tb = qf_winners[i+1]
        pa, pb, wid = predict_match(ta, tb, "semi_finals", "2026-07-14")
        rounds["semi_finals"].append({
            "team1": team_info(ta),
            "team2": team_info(tb),
            "team1_prob": round(pa*100, 1),
            "team2_prob": round(pb*100, 1),
            "winner_id": wid,
        })
        sf_winners.append(ta if wid == ta["id"] else tb)

    # --- Final ---
    ta = sf_winners[0]; tb = sf_winners[1]
    pa, pb, wid = predict_match(ta, tb, "final", "2026-07-19")
    rounds["final"].append({
        "team1": team_info(ta),
        "team2": team_info(tb),
        "team1_prob": round(pa*100, 1),
        "team2_prob": round(pb*100, 1),
        "winner_id": wid,
    })
    champion = ta if wid == ta["id"] else tb

    return {
        "model": "XGBoost (dual-prediction symmetrized)",
        "champion": team_info(champion),
        "rounds": rounds,
    }


@app.get("/api/bracket/v3")
def bracket_v3():
    """Simulate knockout bracket using Logistic Regression dual-prediction.
    
    Same logic as bracket/v2 but uses L2-regularized logistic regression
    instead of XGBoost. More honest probabilities, less overfitting.
    """
    teams_raw = load_teams_raw()
    teams_lookup = {t["id"]: t for t in teams_raw}
    matches_enriched = load_matches_enriched()
    matches_raw = load_matches_raw()

    team_history = build_team_history(matches_raw, teams_lookup)

    lr = get_logistic_predictor()

    def predict_match(team_a, team_b, stage, date):
        aid = team_a["id"]; bid = team_b["id"]
        h_a = team_history.get(aid, []); h_b = team_history.get(bid, [])
        try:
            fwd = compute_match_features(aid, bid, team_a, team_b, h_a, h_b, stage, date)
            rev = compute_match_features(bid, aid, team_b, team_a, h_b, h_a, stage, date)
            pf = lr.predict(fwd); pr = lr.predict(rev)
            pa = round((pf + (1.0 - pr)) / 2.0, 4)
            pb = round(1.0 - pa, 4)
            return pa, pb, (aid if pa >= 0.5 else bid)
        except Exception:
            return 0.5, 0.5, aid

    r16_matches = [m for m in matches_enriched if m.get("stage") == "round_of_16"]
    r16_matches.sort(key=lambda m: m.get("id", 0))

    rounds = {"round_of_16": [], "quarter_finals": [], "semi_finals": [], "final": []}

    r16_winners = []
    for m in r16_matches:
        t1id = m.get("team1_id"); t2id = m.get("team2_id")
        t1 = teams_lookup.get(t1id, {}); t2 = teams_lookup.get(t2id, {})
        # Use logistic regression live for R16 too
        try:
            pa, pb, wid = predict_match(t1, t2, "round_of_16", m.get("date", "2026-07-05"))
        except:
            pa, pb = 50, 50
            wid = t1id
        winner_id = t1id if pa >= pb else t2id
        rounds["round_of_16"].append({
            "match_id": m.get("id"),
            "team1": team_info(t1),
            "team2": team_info(t2),
            "team1_prob": round(pa*100, 1),
            "team2_prob": round(pb*100, 1),
            "winner_id": winner_id,
        })
        r16_winners.append(teams_lookup.get(winner_id, teams_lookup.get(t1id)))

    qf_winners = []
    for i in range(0, 8, 2):
        ta = r16_winners[i]; tb = r16_winners[i+1]
        pa, pb, wid = predict_match(ta, tb, "quarter_finals", "2026-07-10")
        rounds["quarter_finals"].append({
            "team1": team_info(ta), "team2": team_info(tb),
            "team1_prob": round(pa*100, 1), "team2_prob": round(pb*100, 1),
            "winner_id": wid,
        })
        qf_winners.append(ta if wid == ta["id"] else tb)

    sf_winners = []
    for i in range(0, 4, 2):
        ta = qf_winners[i]; tb = qf_winners[i+1]
        pa, pb, wid = predict_match(ta, tb, "semi_finals", "2026-07-14")
        rounds["semi_finals"].append({
            "team1": team_info(ta), "team2": team_info(tb),
            "team1_prob": round(pa*100, 1), "team2_prob": round(pb*100, 1),
            "winner_id": wid,
        })
        sf_winners.append(ta if wid == ta["id"] else tb)

    ta = sf_winners[0]; tb = sf_winners[1]
    pa, pb, wid = predict_match(ta, tb, "final", "2026-07-19")
    rounds["final"].append({
        "team1": team_info(ta), "team2": team_info(tb),
        "team1_prob": round(pa*100, 1), "team2_prob": round(pb*100, 1),
        "winner_id": wid,
    })
    champion = ta if wid == ta["id"] else tb

    return {
        "model": "Logistic Regression (L2, C=0.1, dual-prediction symmetrized)",
        "champion": team_info(champion),
        "rounds": rounds,
    }


def team_info(t: dict) -> dict:
    return {
        "id": t.get("id"),
        "name": t.get("name", "TBD"),
        "flag": t.get("flag_emoji", "🏳️"),
        "flag_code": t.get("flag_code", ""),
        "elo": t.get("elo_rating", 0),
        "fifa_rank": t.get("fifa_rank", 0),
    }


# Serve frontend SPA — must be last (after all /api routes)
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
