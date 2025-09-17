#!/usr/bin/env python3
"""
Auto Configure InfluxDB and Grafana
Automatically retrieves tokens and configures the system
"""

import os
import sys
import json
import requests
import subprocess
import time

def get_influxdb_token():
    """Try to retrieve the InfluxDB token using admin credentials"""
    print("🔑 Attempting to retrieve InfluxDB token...")
    
    # First, try to sign in and get the auth cookie
    session = requests.Session()
    
    # Try to sign in with the credentials we set
    signin_url = "http://localhost:8086/api/v2/signin"
    signin_data = {
        "username": "admin",
        "password": "supersecret123"
    }
    
    try:
        response = session.post(signin_url, json=signin_data)
        if response.status_code == 200 or response.status_code == 204:
            print("✅ Successfully signed in to InfluxDB")
            
            # Now get the list of authorizations/tokens
            auth_url = "http://localhost:8086/api/v2/authorizations"
            auth_response = session.get(auth_url)
            
            if auth_response.status_code == 200:
                auth_data = auth_response.json()
                if auth_data.get("authorizations"):
                    # Get the first token with write permissions
                    for auth in auth_data["authorizations"]:
                        if auth.get("status") == "active":
                            token = auth.get("token")
                            print(f"✅ Retrieved token: {token[:20]}...")
                            return token
            
            # If no tokens found, create one
            print("📝 Creating new token...")
            create_token_data = {
                "description": "Agent System Token",
                "orgID": None,  # Will be filled
                "permissions": []
            }
            
            # Get org ID first
            org_response = session.get("http://localhost:8086/api/v2/orgs")
            if org_response.status_code == 200:
                orgs = org_response.json().get("orgs", [])
                for org in orgs:
                    if org["name"] == "agent-system":
                        org_id = org["id"]
                        create_token_data["orgID"] = org_id
                        
                        # Add read/write permissions for all resources
                        resources = ["authorizations", "buckets", "dashboards", "orgs", "sources", 
                                   "tasks", "telegrafs", "users", "variables", "scrapers", "secrets",
                                   "labels", "views", "documents", "notificationRules", 
                                   "notificationEndpoints", "checks", "dbrp", "notebooks", "annotations",
                                   "remotes", "replications"]
                        
                        for resource in resources:
                            create_token_data["permissions"].append({
                                "action": "read",
                                "resource": {
                                    "type": resource,
                                    "orgID": org_id
                                }
                            })
                            create_token_data["permissions"].append({
                                "action": "write",
                                "resource": {
                                    "type": resource,
                                    "orgID": org_id
                                }
                            })
                        
                        token_response = session.post("http://localhost:8086/api/v2/authorizations", 
                                                     json=create_token_data)
                        if token_response.status_code in [200, 201]:
                            token_data = token_response.json()
                            token = token_data.get("token")
                            print(f"✅ Created new token: {token[:20]}...")
                            return token
        else:
            print(f"❌ Failed to sign in: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

def update_env_file(token):
    """Update the .env.influxdb file with the actual token"""
    env_file = ".env.influxdb"
    
    # Read existing content
    lines = []
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
    # Update or add the token line
    token_found = False
    for i, line in enumerate(lines):
        if line.startswith("INFLUXDB_TOKEN="):
            lines[i] = f"INFLUXDB_TOKEN={token}\n"
            token_found = True
            break
    
    if not token_found:
        lines.append(f"INFLUXDB_TOKEN={token}\n")
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Updated {env_file} with actual token")
    
    # Also set it as environment variable for current session
    os.environ["INFLUXDB_TOKEN"] = token

def configure_grafana_datasource(token):
    """Configure Grafana to use InfluxDB with the correct token"""
    print("\n📊 Configuring Grafana data source...")
    
    session = requests.Session()
    session.auth = ("admin", "admin123")
    
    # First, delete any existing InfluxDB data sources
    try:
        # Get existing data sources
        ds_response = session.get("http://localhost:3000/api/datasources")
        if ds_response.status_code == 200:
            datasources = ds_response.json()
            for ds in datasources:
                if ds.get("type") == "influxdb":
                    print(f"🗑️  Removing old datasource: {ds['name']}")
                    session.delete(f"http://localhost:3000/api/datasources/{ds['id']}")
    except:
        pass
    
    # Create new InfluxDB datasource
    datasource_config = {
        "name": "InfluxDB",
        "type": "influxdb",
        "url": "http://influxdb:8086",
        "access": "proxy",
        "isDefault": True,
        "jsonData": {
            "version": "Flux",
            "organization": "agent-system",
            "defaultBucket": "performance_metrics",
            "tlsSkipVerify": True
        },
        "secureJsonData": {
            "token": token
        }
    }
    
    try:
        response = session.post(
            "http://localhost:3000/api/datasources",
            json=datasource_config
        )
        
        if response.status_code in [200, 201]:
            print("✅ Grafana data source configured successfully")
            
            # Test the connection
            ds_data = response.json()
            test_response = session.post(
                f"http://localhost:3000/api/datasources/{ds_data['id']}/resources/test"
            )
            if test_response.status_code == 200:
                print("✅ Data source connection test successful")
            return True
        else:
            print(f"⚠️  Failed to configure Grafana: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error configuring Grafana: {e}")
    
    return False

def test_write_data(token):
    """Test writing data to InfluxDB with the retrieved token"""
    print("\n📝 Testing data write with retrieved token...")
    
    try:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        
        client = InfluxDBClient(
            url="http://localhost:8086",
            token=token,
            org="agent-system"
        )
        
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Write test point
        point = Point("performance_metrics") \
            .tag("host", "agent-system") \
            .tag("metric_type", "cpu") \
            .field("value", 42.5)
        
        write_api.write(bucket="performance_metrics", org="agent-system", record=point)
        print("✅ Successfully wrote test data to InfluxDB!")
        
        client.close()
        return True
    except Exception as e:
        print(f"❌ Failed to write test data: {e}")
        return False

def main():
    print("🚀 Auto-configuring InfluxDB and Grafana")
    print("=" * 60)
    
    # Step 1: Get the InfluxDB token
    token = get_influxdb_token()
    if not token:
        print("\n❌ Could not retrieve InfluxDB token")
        print("Please ensure InfluxDB is running and initial setup is complete")
        return False
    
    # Step 2: Update environment file
    update_env_file(token)
    
    # Step 3: Configure Grafana
    if configure_grafana_datasource(token):
        print("✅ Grafana configured with InfluxDB token")
    
    # Step 4: Test writing data
    if test_write_data(token):
        print("\n✅ System fully configured and tested!")
        print("\n📊 Next steps:")
        print("1. Run: python start_monitoring.py")
        print("2. Visit Grafana: http://localhost:3000")
        print("3. View dashboards with real data!")
        return True
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)