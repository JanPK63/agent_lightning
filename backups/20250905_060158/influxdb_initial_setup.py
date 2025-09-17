#!/usr/bin/env python3
"""
InfluxDB Initial Setup
Performs the initial setup of InfluxDB via API
"""

import requests
import json
import time
import os

def setup_influxdb():
    """Perform initial setup of InfluxDB"""
    
    # Check if setup is needed
    setup_check_url = "http://localhost:8086/api/v2/setup"
    response = requests.get(setup_check_url)
    
    if response.status_code == 200:
        setup_status = response.json()
        if setup_status.get("allowed") == True:
            print("📋 InfluxDB needs initial setup")
            
            # Perform the setup
            setup_url = "http://localhost:8086/api/v2/setup"
            setup_data = {
                "username": "admin",
                "password": "supersecret123",
                "org": "agent-system",
                "bucket": "performance_metrics",
                "retentionPeriodSeconds": 2592000,  # 30 days
                "token": "agent-system-token-supersecret-12345678"
            }
            
            print("🔧 Performing initial setup...")
            print(f"  Organization: {setup_data['org']}")
            print(f"  Bucket: {setup_data['bucket']}")
            print(f"  Username: {setup_data['username']}")
            
            response = requests.post(setup_url, json=setup_data)
            
            if response.status_code == 201 or response.status_code == 200:
                print("✅ Initial setup completed successfully!")
                
                # Save the token to environment file
                env_file = ".env.influxdb"
                with open(env_file, 'w') as f:
                    f.write(f"INFLUXDB_URL=http://localhost:8086\n")
                    f.write(f"INFLUXDB_TOKEN={setup_data['token']}\n")
                    f.write(f"INFLUXDB_ORG={setup_data['org']}\n")
                    f.write(f"INFLUXDB_BUCKET={setup_data['bucket']}\n")
                
                print(f"💾 Configuration saved to {env_file}")
                
                # Also update the system environment for current session
                os.environ['INFLUXDB_URL'] = 'http://localhost:8086'
                os.environ['INFLUXDB_TOKEN'] = setup_data['token']
                os.environ['INFLUXDB_ORG'] = setup_data['org']
                os.environ['INFLUXDB_BUCKET'] = setup_data['bucket']
                
                return True
            else:
                print(f"❌ Setup failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
        else:
            print("✅ InfluxDB is already set up")
            
            # Try to sign in and get configuration
            signin_url = "http://localhost:8086/api/v2/signin"
            signin_data = {
                "username": "admin",
                "password": "supersecret123"
            }
            
            session = requests.Session()
            response = session.post(signin_url, json=signin_data)
            
            if response.status_code in [200, 204]:
                print("✅ Successfully signed in to InfluxDB")
                
                # Get organization info
                orgs_response = session.get("http://localhost:8086/api/v2/orgs")
                if orgs_response.status_code == 200:
                    orgs = orgs_response.json().get("orgs", [])
                    if orgs:
                        org = orgs[0]
                        print(f"  Organization: {org['name']} (ID: {org['id']})")
                
                # Get buckets
                buckets_response = session.get("http://localhost:8086/api/v2/buckets")
                if buckets_response.status_code == 200:
                    buckets = buckets_response.json().get("buckets", [])
                    print("  Buckets:")
                    for bucket in buckets:
                        if not bucket['name'].startswith('_'):
                            print(f"    - {bucket['name']}")
                
                return True
            else:
                print(f"❌ Could not sign in: {response.status_code}")
                return False
    else:
        print(f"❌ Could not check setup status: {response.status_code}")
        return False

if __name__ == "__main__":
    print("🚀 InfluxDB Initial Setup")
    print("=" * 60)
    
    # Wait a moment to ensure InfluxDB is ready
    time.sleep(2)
    
    if setup_influxdb():
        print("\n✅ Setup completed successfully!")
        print("\nYou can now:")
        print("1. Access InfluxDB at http://localhost:8086")
        print("2. Use the API with the configured token")
        print("3. Start sending metrics to the system")
    else:
        print("\n❌ Setup failed. Please check the logs and try again.")