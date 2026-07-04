import json

with open('backend/data/teams.json') as f:
    teams = json.load(f)

targets = ['England', 'France', 'Brazil', 'Argentina', 'Spain', 'Norway', 'Curacao', 'Turkiye']
for t in teams:
    if t['name'] in targets:
        sv = t.get('squad_value_millions', 0)
        print(f"{t['name']:20s} FIFA#{t.get('fifa_rank','?')}  EUR {sv}M")
