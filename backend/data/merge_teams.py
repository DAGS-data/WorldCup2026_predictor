import json
from pathlib import Path

data_dir = Path(__file__).parent

teams = []
for filename in ("teams_a.json", "teams_b.json", "teams_c.json"):
    with open(data_dir / filename) as f:
        teams.extend(json.load(f))

teams.sort(key=lambda t: (t["group"], t["id"]))

assert len(teams) == 48, f"Expected 48 teams, got {len(teams)}"

with open(data_dir / "teams.json", "w") as f:
    json.dump(teams, f, indent=2, ensure_ascii=False)

print(f"Written {len(teams)} teams to teams.json")
