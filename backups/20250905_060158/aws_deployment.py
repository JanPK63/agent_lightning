#!/usr/bin/env python3
"""
AWS Deployment Configuration for Agent Code Executor
Enables agents to deploy code to AWS EC2 instances
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import boto3
from code_executor import CodeExecutor, ExecutionConfig, ExecutionMode, AgentCodeExecutor


@dataclass 
class AWSConfig:
    """AWS deployment configuration"""
    region: str = "us-east-1"
    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    key_name: Optional[str] = None
    key_path: Optional[str] = None
    security_group: Optional[str] = None
    instance_type: str = "t2.micro"
    ami_id: str = "ami-0c02fb55731490381"  # Amazon Linux 2


class AWSDeploymentManager:
    """Manage deployments to AWS EC2 instances"""
    
    def __init__(self, aws_config: AWSConfig):
        self.config = aws_config
        self.ec2_client = boto3.client('ec2', region_name=self.config.region)
        self.ec2_resource = boto3.resource('ec2', region_name=self.config.region)
        self.ssm_client = boto3.client('ssm', region_name=self.config.region)
    
    def get_or_create_instance(self) -> Dict[str, Any]:
        """Get existing instance or create a new one"""
        
        if self.config.instance_id:
            # Use existing instance
            instance = self.ec2_resource.Instance(self.config.instance_id)
            if instance.state['Name'] != 'running':
                instance.start()
                instance.wait_until_running()
            
            return {
                "instance_id": instance.id,
                "public_ip": instance.public_ip_address,
                "private_ip": instance.private_ip_address,
                "state": instance.state['Name']
            }
        else:
            # Create new instance
            return self.create_instance()
    
    def create_instance(self) -> Dict[str, Any]:
        """Create a new EC2 instance for deployment"""
        
        # Create key pair if not exists
        if not self.config.key_name:
            self.config.key_name = "agent-lightning-key"
            self.create_key_pair()
        
        # Create security group if not exists
        if not self.config.security_group:
            self.config.security_group = self.create_security_group()
        
        # Launch instance
        user_data = """#!/bin/bash
        yum update -y
        yum install -y python3 python3-pip git docker
        systemctl start docker
        usermod -a -G docker ec2-user
        pip3 install fastapi uvicorn paramiko gitpython boto3
        """
        
        response = self.ec2_client.run_instances(
            ImageId=self.config.ami_id,
            InstanceType=self.config.instance_type,
            KeyName=self.config.key_name,
            SecurityGroups=[self.config.security_group],
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': 'agent-lightning-deployment'},
                    {'Key': 'ManagedBy', 'Value': 'AgentLightning'}
                ]
            }]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        
        # Wait for instance to be running
        waiter = self.ec2_client.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # Get instance details
        instance = self.ec2_resource.Instance(instance_id)
        
        return {
            "instance_id": instance.id,
            "public_ip": instance.public_ip_address,
            "private_ip": instance.private_ip_address,
            "state": instance.state['Name'],
            "key_name": self.config.key_name
        }
    
    def create_key_pair(self):
        """Create SSH key pair for EC2 access"""
        
        try:
            response = self.ec2_client.create_key_pair(KeyName=self.config.key_name)
            
            # Save private key
            key_path = Path.home() / '.ssh' / f"{self.config.key_name}.pem"
            key_path.parent.mkdir(exist_ok=True)
            key_path.write_text(response['KeyMaterial'])
            key_path.chmod(0o400)
            
            self.config.key_path = str(key_path)
            print(f"✅ Created key pair: {self.config.key_name}")
            print(f"   Private key saved: {key_path}")
            
        except self.ec2_client.exceptions.ClientError as e:
            if 'InvalidKeyPair.Duplicate' in str(e):
                print(f"Key pair {self.config.key_name} already exists")
                self.config.key_path = str(Path.home() / '.ssh' / f"{self.config.key_name}.pem")
    
    def create_security_group(self) -> str:
        """Create security group for deployment"""
        
        sg_name = "agent-lightning-sg"
        
        try:
            response = self.ec2_client.create_security_group(
                GroupName=sg_name,
                Description='Security group for Agent Lightning deployments'
            )
            
            security_group_id = response['GroupId']
            
            # Add rules
            self.ec2_client.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 80,
                        'ToPort': 80,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 443,
                        'ToPort': 443,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 8000,
                        'ToPort': 9000,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            print(f"✅ Created security group: {sg_name}")
            return sg_name
            
        except self.ec2_client.exceptions.ClientError as e:
            if 'InvalidGroup.Duplicate' in str(e):
                print(f"Security group {sg_name} already exists")
                return sg_name
            raise
    
    def get_executor_config(self, instance_info: Dict[str, Any]) -> ExecutionConfig:
        """Get execution config for the AWS instance"""
        
        return ExecutionConfig(
            mode=ExecutionMode.REMOTE_SSH,
            working_directory="/home/ec2-user/agent-lightning",
            ssh_host=instance_info["public_ip"],
            ssh_user="ec2-user",
            ssh_key_path=self.config.key_path
        )
    
    def terminate_instance(self, instance_id: str):
        """Terminate an EC2 instance"""
        
        instance = self.ec2_resource.Instance(instance_id)
        instance.terminate()
        print(f"✅ Terminated instance: {instance_id}")


# Configuration for your Ubuntu server
def get_ubuntu_server_config(
    server_ip: str,
    username: str = "ubuntu",
    key_path: Optional[str] = None
) -> ExecutionConfig:
    """Get execution config for Ubuntu server"""
    
    return ExecutionConfig(
        mode=ExecutionMode.REMOTE_SSH,
        working_directory=f"/home/{username}/agent-lightning",
        ssh_host=server_ip,
        ssh_user=username,
        ssh_key_path=key_path
    )


# Example usage
async def deploy_to_aws_example():
    """Example of deploying code to AWS"""
    
    # Setup AWS deployment
    aws_config = AWSConfig(
        region="us-east-1",
        key_name="my-agent-key"
    )
    
    manager = AWSDeploymentManager(aws_config)
    instance_info = manager.get_or_create_instance()
    
    print(f"🚀 Deploying to AWS instance: {instance_info['instance_id']}")
    print(f"   Public IP: {instance_info['public_ip']}")
    
    # Get executor for the AWS instance
    exec_config = manager.get_executor_config(instance_info)
    executor = CodeExecutor(exec_config)
    agent_executor = AgentCodeExecutor(executor)
    
    # Deploy a simple web application
    code_snippets = {
        "app.py": """from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Agent Lightning on AWS!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
        "requirements.txt": "fastapi\nuvicorn\n",
        "deploy.sh": """#!/bin/bash
pip3 install -r requirements.txt
nohup python3 app.py > app.log 2>&1 &
echo "Application deployed and running on port 8000"
"""
    }
    
    # Implement the feature
    result = await agent_executor.implement_feature(
        "Deploy FastAPI application to AWS",
        code_snippets
    )
    
    if result["success"]:
        # Make deploy script executable and run it
        await executor.execute_command("chmod +x deploy.sh")
        deploy_result = await executor.execute_command("./deploy.sh")
        
        print(f"✅ Deployment result: {deploy_result.output}")
        print(f"🌐 Application available at: http://{instance_info['public_ip']}:8000")
    
    executor.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(deploy_to_aws_example())