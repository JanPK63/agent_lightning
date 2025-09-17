#!/usr/bin/env python3
import requests
import json

def fix_dashboard_layout(uid):
    # Get dashboard
    resp = requests.get(f"http://admin:admin@localhost:3000/api/dashboards/uid/{uid}")
    if resp.status_code != 200:
        return False
    
    data = resp.json()
    dashboard = data['dashboard']
    
    # Fix panel layout - full width, stacked vertically
    if 'panels' in dashboard:
        for i, panel in enumerate(dashboard['panels']):
            panel['gridPos'] = {
                'h': 8,
                'w': 24, 
                'x': 0,
                'y': i * 8
            }
    
    # Update dashboard
    update_data = {
        'dashboard': dashboard,
        'overwrite': True
    }
    
    resp = requests.post(
        "http://admin:admin@localhost:3000/api/dashboards/db",
        json=update_data
    )
    return resp.status_code == 200

# Get all dashboards and fix them
resp = requests.get("http://admin:admin@localhost:3000/api/search")
dashboards = resp.json()

for dash in dashboards:
    uid = dash.get('uid')
    if uid:
        print(f"Fixing {dash.get('title', uid)}")
        fix_dashboard_layout(uid)

print("All dashboards updated to full-width vertical layout")