#!/usr/bin/env python3
"""
Quick SSH test to verify connection to 13.38.102.28
"""

import paramiko
import os

def test_ssh_connection():
    try:
        # Test SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Try connecting with common key locations
        key_paths = [
            os.path.expanduser("~/.ssh/id_rsa"),
            os.path.expanduser("~/.ssh/id_ed25519"),
            "/Users/jankootstra/.ssh/id_rsa"
        ]
        
        for key_path in key_paths:
            if os.path.exists(key_path):
                print(f"Trying key: {key_path}")
                try:
                    ssh.connect(
                        hostname="13.38.102.28",
                        username="ubuntu",
                        key_filename=key_path,
                        timeout=10
                    )
                    
                    # Test command
                    stdin, stdout, stderr = ssh.exec_command("ls -la /home")
                    output = stdout.read().decode()
                    error = stderr.read().decode()
                    
                    print(f"✅ SSH Connection successful with {key_path}")
                    print(f"Output: {output}")
                    if error:
                        print(f"Error: {error}")
                    
                    # Check for digital-identity-frontend
                    stdin, stdout, stderr = ssh.exec_command("ls -la /home/digital-identity-frontend")
                    output = stdout.read().decode()
                    error = stderr.read().decode()
                    
                    if "No such file" in error:
                        print("❌ /home/digital-identity-frontend not found")
                        # Try other locations
                        for path in ["/home/ubuntu/digital-identity-frontend", "/var/www/digital-identity-frontend", "/opt/digital-identity-frontend"]:
                            stdin, stdout, stderr = ssh.exec_command(f"ls -la {path}")
                            output = stdout.read().decode()
                            error = stderr.read().decode()
                            if "No such file" not in error:
                                print(f"✅ Found project at: {path}")
                                break
                    else:
                        print(f"✅ Found /home/digital-identity-frontend: {output}")
                    
                    ssh.close()
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed with {key_path}: {e}")
                    continue
        
        print("❌ All SSH key attempts failed")
        return False
        
    except Exception as e:
        print(f"❌ SSH test failed: {e}")
        return False

if __name__ == "__main__":
    test_ssh_connection()