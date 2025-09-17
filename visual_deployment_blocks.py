#!/usr/bin/env python3
"""
Visual Docker/K8s Deployment Blocks for Agent Lightning
Visual blocks for creating and managing containerized deployments
"""

import os
import sys
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import uuid
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualBlock,
    BlockType,
    VisualProgram,
    BlockFactory
)


class DeploymentType(Enum):
    """Types of deployments"""
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker-compose"
    KUBERNETES = "kubernetes"
    HELM = "helm"
    KUBERNETES_OPERATOR = "operator"


class ResourceType(Enum):
    """Kubernetes resource types"""
    DEPLOYMENT = "Deployment"
    SERVICE = "Service"
    INGRESS = "Ingress"
    CONFIGMAP = "ConfigMap"
    SECRET = "Secret"
    PERSISTENTVOLUME = "PersistentVolume"
    PERSISTENTVOLUMECLAIM = "PersistentVolumeClaim"
    STATEFULSET = "StatefulSet"
    DAEMONSET = "DaemonSet"
    JOB = "Job"
    CRONJOB = "CronJob"
    NAMESPACE = "Namespace"
    SERVICEACCOUNT = "ServiceAccount"
    ROLE = "Role"
    ROLEBINDING = "RoleBinding"


@dataclass
class DockerConfig:
    """Docker container configuration"""
    image: str = "alpine:latest"
    tag: str = "latest"
    ports: List[str] = field(default_factory=list)  # ["80:8080", "443:8443"]
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)  # ["/host:/container"]
    command: Optional[str] = None
    entrypoint: Optional[str] = None
    working_dir: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    networks: List[str] = field(default_factory=list)
    restart_policy: str = "always"  # no, on-failure, always, unless-stopped
    cpu_limit: Optional[str] = None  # "0.5" or "500m"
    memory_limit: Optional[str] = None  # "512M"
    health_check: Optional[Dict[str, Any]] = None


@dataclass
class KubernetesConfig:
    """Kubernetes resource configuration"""
    api_version: str = "apps/v1"
    kind: str = "Deployment"
    name: str = "app"
    namespace: str = "default"
    replicas: int = 1
    selector: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)


class DeploymentBlockFactory:
    """Factory for creating deployment-related visual blocks"""
    
    @staticmethod
    def create_dockerfile_block() -> VisualBlock:
        """Create a Dockerfile generation block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="Dockerfile Generator"
        )
        block.properties = {
            "base_image": "python:3.9-slim",
            "workdir": "/app",
            "copy_files": [".", "/app"],
            "run_commands": ["pip install -r requirements.txt"],
            "expose_ports": [8000],
            "cmd": ["python", "app.py"],
            "env_vars": {},
            "labels": {},
            "user": None,
            "healthcheck": None
        }
        block.properties["deployment_type"] = "docker"
        block.properties["generates"] = "Dockerfile"
        return block
    
    @staticmethod
    def create_docker_build_block() -> VisualBlock:
        """Create a Docker build block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="Docker Build"
        )
        block.properties = {
            "dockerfile_path": "./Dockerfile",
            "context_path": ".",
            "image_name": "myapp",
            "tag": "latest",
            "build_args": {},
            "target": None,
            "cache": True,
            "push": False,
            "registry": None
        }
        block.properties["deployment_type"] = "docker"
        block.properties["action"] = "build"
        return block
    
    @staticmethod
    def create_docker_run_block() -> VisualBlock:
        """Create a Docker run block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="Docker Run"
        )
        config = DockerConfig()
        block.properties = {
            "image": config.image,
            "name": "mycontainer",
            "ports": config.ports,
            "environment": config.environment,
            "volumes": config.volumes,
            "detach": True,
            "remove": False,
            "restart": config.restart_policy,
            "network": None,
            "command": config.command
        }
        block.properties["deployment_type"] = "docker"
        block.properties["action"] = "run"
        return block
    
    @staticmethod
    def create_docker_compose_block() -> VisualBlock:
        """Create a Docker Compose configuration block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="Docker Compose"
        )
        block.properties = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx:alpine",
                    "ports": ["80:80"],
                    "volumes": ["./html:/usr/share/nginx/html"]
                },
                "api": {
                    "build": ".",
                    "ports": ["8000:8000"],
                    "environment": {
                        "DATABASE_URL": "postgresql://localhost/db"
                    }
                }
            },
            "networks": {},
            "volumes": {},
            "secrets": {},
            "configs": {}
        }
        block.properties["deployment_type"] = "docker-compose"
        block.properties["generates"] = "docker-compose.yml"
        return block
    
    @staticmethod
    def create_k8s_deployment_block() -> VisualBlock:
        """Create a Kubernetes Deployment block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="K8s Deployment"
        )
        block.properties = {
            "name": "app-deployment",
            "namespace": "default",
            "replicas": 3,
            "image": "myapp:latest",
            "ports": [{"containerPort": 8080}],
            "env": [],
            "resources": {
                "requests": {"memory": "128Mi", "cpu": "250m"},
                "limits": {"memory": "256Mi", "cpu": "500m"}
            },
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxSurge": 1,
                    "maxUnavailable": 0
                }
            },
            "livenessProbe": None,
            "readinessProbe": None
        }
        block.properties["deployment_type"] = "kubernetes"
        block.properties["resource_type"] = ResourceType.DEPLOYMENT.value
        return block
    
    @staticmethod
    def create_k8s_service_block() -> VisualBlock:
        """Create a Kubernetes Service block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="K8s Service"
        )
        block.properties = {
            "name": "app-service",
            "namespace": "default",
            "type": "ClusterIP",  # ClusterIP, NodePort, LoadBalancer
            "ports": [
                {
                    "port": 80,
                    "targetPort": 8080,
                    "protocol": "TCP"
                }
            ],
            "selector": {"app": "myapp"},
            "sessionAffinity": "None"
        }
        block.properties["deployment_type"] = "kubernetes"
        block.properties["resource_type"] = ResourceType.SERVICE.value
        return block
    
    @staticmethod
    def create_k8s_ingress_block() -> VisualBlock:
        """Create a Kubernetes Ingress block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="K8s Ingress"
        )
        block.properties = {
            "name": "app-ingress",
            "namespace": "default",
            "ingressClassName": "nginx",
            "rules": [
                {
                    "host": "app.example.com",
                    "paths": [
                        {
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": "app-service",
                                "port": 80
                            }
                        }
                    ]
                }
            ],
            "tls": [],
            "annotations": {
                "nginx.ingress.kubernetes.io/rewrite-target": "/"
            }
        }
        block.properties["deployment_type"] = "kubernetes"
        block.properties["resource_type"] = ResourceType.INGRESS.value
        return block
    
    @staticmethod
    def create_k8s_configmap_block() -> VisualBlock:
        """Create a Kubernetes ConfigMap block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="K8s ConfigMap"
        )
        block.properties = {
            "name": "app-config",
            "namespace": "default",
            "data": {
                "config.yaml": "key: value",
                "app.properties": "property=value"
            },
            "binaryData": {}
        }
        block.properties["deployment_type"] = "kubernetes"
        block.properties["resource_type"] = ResourceType.CONFIGMAP.value
        return block
    
    @staticmethod
    def create_k8s_secret_block() -> VisualBlock:
        """Create a Kubernetes Secret block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="K8s Secret"
        )
        block.properties = {
            "name": "app-secret",
            "namespace": "default",
            "type": "Opaque",  # Opaque, kubernetes.io/tls, kubernetes.io/dockerconfigjson
            "data": {},  # base64 encoded
            "stringData": {  # plain text, will be encoded
                "username": "admin",
                "password": "secret"
            }
        }
        block.properties["deployment_type"] = "kubernetes"
        block.properties["resource_type"] = ResourceType.SECRET.value
        return block
    
    @staticmethod
    def create_k8s_statefulset_block() -> VisualBlock:
        """Create a Kubernetes StatefulSet block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="K8s StatefulSet"
        )
        block.properties = {
            "name": "database",
            "namespace": "default",
            "serviceName": "database-service",
            "replicas": 3,
            "image": "postgres:13",
            "volumeClaimTemplates": [
                {
                    "name": "data",
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {
                        "requests": {"storage": "10Gi"}
                    }
                }
            ],
            "podManagementPolicy": "OrderedReady",
            "updateStrategy": {
                "type": "RollingUpdate"
            }
        }
        block.properties["deployment_type"] = "kubernetes"
        block.properties["resource_type"] = ResourceType.STATEFULSET.value
        return block
    
    @staticmethod
    def create_helm_chart_block() -> VisualBlock:
        """Create a Helm Chart block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="Helm Chart"
        )
        block.properties = {
            "name": "myapp",
            "version": "0.1.0",
            "appVersion": "1.0.0",
            "description": "A Helm chart for my application",
            "values": {
                "replicaCount": 1,
                "image": {
                    "repository": "myapp",
                    "tag": "latest",
                    "pullPolicy": "IfNotPresent"
                },
                "service": {
                    "type": "ClusterIP",
                    "port": 80
                },
                "ingress": {
                    "enabled": False
                }
            },
            "dependencies": []
        }
        block.properties["deployment_type"] = "helm"
        block.properties["generates"] = "helm-chart"
        return block


class DeploymentGenerator:
    """Generate deployment configurations from visual blocks"""
    
    def __init__(self):
        self.output_dir = Path("deployments")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_dockerfile(self, block: VisualBlock) -> str:
        """Generate Dockerfile from visual block"""
        props = block.properties
        
        dockerfile = f"FROM {props.get('base_image', 'alpine:latest')}\n\n"
        
        # Set working directory
        if props.get('workdir'):
            dockerfile += f"WORKDIR {props['workdir']}\n\n"
        
        # Environment variables
        env_vars = props.get('env_vars', {})
        for key, value in env_vars.items():
            dockerfile += f"ENV {key}={value}\n"
        if env_vars:
            dockerfile += "\n"
        
        # Labels
        labels = props.get('labels', {})
        for key, value in labels.items():
            dockerfile += f"LABEL {key}=\"{value}\"\n"
        if labels:
            dockerfile += "\n"
        
        # Copy files
        copy_files = props.get('copy_files', [])
        if len(copy_files) >= 2:
            dockerfile += f"COPY {copy_files[0]} {copy_files[1]}\n\n"
        
        # Run commands
        run_commands = props.get('run_commands', [])
        for cmd in run_commands:
            dockerfile += f"RUN {cmd}\n"
        if run_commands:
            dockerfile += "\n"
        
        # Expose ports
        expose_ports = props.get('expose_ports', [])
        for port in expose_ports:
            dockerfile += f"EXPOSE {port}\n"
        if expose_ports:
            dockerfile += "\n"
        
        # User
        if props.get('user'):
            dockerfile += f"USER {props['user']}\n\n"
        
        # Health check
        if props.get('healthcheck'):
            hc = props['healthcheck']
            dockerfile += f"HEALTHCHECK --interval={hc.get('interval', '30s')} "
            dockerfile += f"--timeout={hc.get('timeout', '3s')} "
            dockerfile += f"--retries={hc.get('retries', '3')} "
            dockerfile += f"CMD {hc.get('cmd', 'curl -f http://localhost/ || exit 1')}\n\n"
        
        # Command
        cmd = props.get('cmd', [])
        if cmd:
            dockerfile += f"CMD {json.dumps(cmd)}\n"
        
        return dockerfile
    
    def generate_docker_compose(self, block: VisualBlock) -> str:
        """Generate docker-compose.yml from visual block"""
        props = block.properties
        
        compose = {
            "version": props.get("version", "3.8"),
            "services": props.get("services", {}),
            "networks": props.get("networks", {}),
            "volumes": props.get("volumes", {}),
            "secrets": props.get("secrets", {}),
            "configs": props.get("configs", {})
        }
        
        # Remove empty sections
        compose = {k: v for k, v in compose.items() if v}
        
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)
    
    def generate_k8s_resource(self, block: VisualBlock) -> str:
        """Generate Kubernetes resource YAML from visual block"""
        resource_type = block.properties.get("resource_type", "Deployment")
        props = block.properties
        
        if resource_type == ResourceType.DEPLOYMENT.value:
            return self._generate_deployment(props)
        elif resource_type == ResourceType.SERVICE.value:
            return self._generate_service(props)
        elif resource_type == ResourceType.INGRESS.value:
            return self._generate_ingress(props)
        elif resource_type == ResourceType.CONFIGMAP.value:
            return self._generate_configmap(props)
        elif resource_type == ResourceType.SECRET.value:
            return self._generate_secret(props)
        elif resource_type == ResourceType.STATEFULSET.value:
            return self._generate_statefulset(props)
        else:
            return ""
    
    def _generate_deployment(self, props: Dict[str, Any]) -> str:
        """Generate Kubernetes Deployment YAML"""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": props.get("name", "app"),
                "namespace": props.get("namespace", "default")
            },
            "spec": {
                "replicas": props.get("replicas", 1),
                "selector": {
                    "matchLabels": {
                        "app": props.get("name", "app")
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": props.get("name", "app")
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": props.get("name", "app"),
                                "image": props.get("image", "nginx:latest"),
                                "ports": props.get("ports", []),
                                "env": props.get("env", []),
                                "resources": props.get("resources", {})
                            }
                        ]
                    }
                }
            }
        }
        
        # Add strategy if specified
        if props.get("strategy"):
            deployment["spec"]["strategy"] = props["strategy"]
        
        # Add probes if specified
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        if props.get("livenessProbe"):
            container["livenessProbe"] = props["livenessProbe"]
        if props.get("readinessProbe"):
            container["readinessProbe"] = props["readinessProbe"]
        
        return yaml.dump(deployment, default_flow_style=False, sort_keys=False)
    
    def _generate_service(self, props: Dict[str, Any]) -> str:
        """Generate Kubernetes Service YAML"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": props.get("name", "service"),
                "namespace": props.get("namespace", "default")
            },
            "spec": {
                "type": props.get("type", "ClusterIP"),
                "ports": props.get("ports", []),
                "selector": props.get("selector", {})
            }
        }
        
        if props.get("sessionAffinity"):
            service["spec"]["sessionAffinity"] = props["sessionAffinity"]
        
        return yaml.dump(service, default_flow_style=False, sort_keys=False)
    
    def _generate_ingress(self, props: Dict[str, Any]) -> str:
        """Generate Kubernetes Ingress YAML"""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": props.get("name", "ingress"),
                "namespace": props.get("namespace", "default"),
                "annotations": props.get("annotations", {})
            },
            "spec": {
                "ingressClassName": props.get("ingressClassName", "nginx"),
                "rules": props.get("rules", [])
            }
        }
        
        if props.get("tls"):
            ingress["spec"]["tls"] = props["tls"]
        
        return yaml.dump(ingress, default_flow_style=False, sort_keys=False)
    
    def _generate_configmap(self, props: Dict[str, Any]) -> str:
        """Generate Kubernetes ConfigMap YAML"""
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": props.get("name", "config"),
                "namespace": props.get("namespace", "default")
            },
            "data": props.get("data", {})
        }
        
        if props.get("binaryData"):
            configmap["binaryData"] = props["binaryData"]
        
        return yaml.dump(configmap, default_flow_style=False, sort_keys=False)
    
    def _generate_secret(self, props: Dict[str, Any]) -> str:
        """Generate Kubernetes Secret YAML"""
        import base64
        
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": props.get("name", "secret"),
                "namespace": props.get("namespace", "default")
            },
            "type": props.get("type", "Opaque")
        }
        
        # Handle stringData (plain text that needs encoding)
        if props.get("stringData"):
            secret["data"] = {}
            for key, value in props["stringData"].items():
                encoded = base64.b64encode(value.encode()).decode()
                secret["data"][key] = encoded
        elif props.get("data"):
            secret["data"] = props["data"]
        
        return yaml.dump(secret, default_flow_style=False, sort_keys=False)
    
    def _generate_statefulset(self, props: Dict[str, Any]) -> str:
        """Generate Kubernetes StatefulSet YAML"""
        statefulset = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": props.get("name", "statefulset"),
                "namespace": props.get("namespace", "default")
            },
            "spec": {
                "serviceName": props.get("serviceName", "service"),
                "replicas": props.get("replicas", 1),
                "selector": {
                    "matchLabels": {
                        "app": props.get("name", "statefulset")
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": props.get("name", "statefulset")
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": props.get("name", "statefulset"),
                                "image": props.get("image", "nginx:latest"),
                                "ports": props.get("ports", [])
                            }
                        ]
                    }
                },
                "volumeClaimTemplates": props.get("volumeClaimTemplates", [])
            }
        }
        
        if props.get("podManagementPolicy"):
            statefulset["spec"]["podManagementPolicy"] = props["podManagementPolicy"]
        
        if props.get("updateStrategy"):
            statefulset["spec"]["updateStrategy"] = props["updateStrategy"]
        
        return yaml.dump(statefulset, default_flow_style=False, sort_keys=False)
    
    def generate_helm_chart(self, block: VisualBlock) -> Dict[str, str]:
        """Generate Helm chart files from visual block"""
        props = block.properties
        chart_name = props.get("name", "myapp")
        
        files = {}
        
        # Chart.yaml
        chart_yaml = {
            "apiVersion": "v2",
            "name": chart_name,
            "version": props.get("version", "0.1.0"),
            "appVersion": props.get("appVersion", "1.0.0"),
            "description": props.get("description", "A Helm chart"),
            "type": "application",
            "dependencies": props.get("dependencies", [])
        }
        files["Chart.yaml"] = yaml.dump(chart_yaml, default_flow_style=False, sort_keys=False)
        
        # values.yaml
        files["values.yaml"] = yaml.dump(props.get("values", {}), default_flow_style=False, sort_keys=False)
        
        # templates/deployment.yaml
        deployment_template = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "chart.fullname" . }}
  labels:
    {{- include "chart.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "chart.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "chart.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: {{ .Values.service.port }}
          protocol: TCP
"""
        files["templates/deployment.yaml"] = deployment_template
        
        # templates/service.yaml
        service_template = """apiVersion: v1
kind: Service
metadata:
  name: {{ include "chart.fullname" . }}
  labels:
    {{- include "chart.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
  - port: {{ .Values.service.port }}
    targetPort: http
    protocol: TCP
    name: http
  selector:
    {{- include "chart.selectorLabels" . | nindent 4 }}
"""
        files["templates/service.yaml"] = service_template
        
        return files


def create_deployment_pipeline() -> VisualProgram:
    """Create a complete deployment pipeline visual program"""
    program = VisualProgram(name="Container Deployment Pipeline")
    factory = DeploymentBlockFactory()
    
    # Create Dockerfile
    dockerfile_block = factory.create_dockerfile_block()
    dockerfile_block.position = (100, 100)
    program.add_block(dockerfile_block)
    
    # Build Docker image
    build_block = factory.create_docker_build_block()
    build_block.position = (300, 100)
    program.add_block(build_block)
    
    # Create K8s Deployment
    deployment_block = factory.create_k8s_deployment_block()
    deployment_block.position = (500, 100)
    program.add_block(deployment_block)
    
    # Create K8s Service
    service_block = factory.create_k8s_service_block()
    service_block.position = (700, 100)
    program.add_block(service_block)
    
    # Create K8s Ingress
    ingress_block = factory.create_k8s_ingress_block()
    ingress_block.position = (900, 100)
    program.add_block(ingress_block)
    
    # Connect blocks
    program.connect_blocks(
        dockerfile_block.block_id, "output",
        build_block.block_id, "input"
    )
    program.connect_blocks(
        build_block.block_id, "output",
        deployment_block.block_id, "input"
    )
    program.connect_blocks(
        deployment_block.block_id, "output",
        service_block.block_id, "input"
    )
    program.connect_blocks(
        service_block.block_id, "output",
        ingress_block.block_id, "input"
    )
    
    return program


def test_deployment_blocks():
    """Test the deployment blocks system"""
    print("\n" + "="*60)
    print("Visual Deployment Blocks Test")
    print("="*60)
    
    factory = DeploymentBlockFactory()
    generator = DeploymentGenerator()
    
    # Test Dockerfile generation
    print("\n🐳 Testing Dockerfile generation:")
    dockerfile_block = factory.create_dockerfile_block()
    dockerfile = generator.generate_dockerfile(dockerfile_block)
    print(dockerfile[:200] + "...")
    with open("deployments/Dockerfile", "w") as f:
        f.write(dockerfile)
    print("   ✅ Generated Dockerfile")
    
    # Test Docker Compose generation
    print("\n🎼 Testing Docker Compose generation:")
    compose_block = factory.create_docker_compose_block()
    compose_yaml = generator.generate_docker_compose(compose_block)
    print(compose_yaml[:200] + "...")
    with open("deployments/docker-compose.yml", "w") as f:
        f.write(compose_yaml)
    print("   ✅ Generated docker-compose.yml")
    
    # Test Kubernetes resources
    print("\n☸️ Testing Kubernetes resource generation:")
    
    # Deployment
    deployment_block = factory.create_k8s_deployment_block()
    deployment_yaml = generator.generate_k8s_resource(deployment_block)
    with open("deployments/deployment.yaml", "w") as f:
        f.write(deployment_yaml)
    print("   ✅ Generated deployment.yaml")
    
    # Service
    service_block = factory.create_k8s_service_block()
    service_yaml = generator.generate_k8s_resource(service_block)
    with open("deployments/service.yaml", "w") as f:
        f.write(service_yaml)
    print("   ✅ Generated service.yaml")
    
    # Ingress
    ingress_block = factory.create_k8s_ingress_block()
    ingress_yaml = generator.generate_k8s_resource(ingress_block)
    with open("deployments/ingress.yaml", "w") as f:
        f.write(ingress_yaml)
    print("   ✅ Generated ingress.yaml")
    
    # Test Helm chart generation
    print("\n⎈ Testing Helm chart generation:")
    helm_block = factory.create_helm_chart_block()
    helm_files = generator.generate_helm_chart(helm_block)
    
    # Create helm chart directory structure
    helm_dir = Path("deployments/myapp-chart")
    helm_dir.mkdir(parents=True, exist_ok=True)
    (helm_dir / "templates").mkdir(exist_ok=True)
    
    for filepath, content in helm_files.items():
        full_path = helm_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    
    print(f"   ✅ Generated Helm chart with {len(helm_files)} files")
    
    # Create complete pipeline
    print("\n🚀 Creating complete deployment pipeline:")
    pipeline = create_deployment_pipeline()
    print(f"   ✅ Created pipeline with {len(pipeline.blocks)} blocks")
    print(f"   ✅ Created {len(pipeline.connections)} connections")
    
    # Export pipeline
    pipeline_json = pipeline.to_json()
    with open("deployments/deployment_pipeline.json", "w") as f:
        f.write(pipeline_json)
    print("   ✅ Exported pipeline to deployment_pipeline.json")
    
    return factory, generator


if __name__ == "__main__":
    print("Visual Docker/K8s Deployment Blocks for Agent Lightning")
    print("="*60)
    
    factory, generator = test_deployment_blocks()
    
    print("\n✅ Deployment Blocks System ready!")
    print("\nFeatures:")
    print("  • Docker container configuration blocks")
    print("  • Docker Compose orchestration")
    print("  • Kubernetes resource blocks (Deployment, Service, Ingress, etc.)")
    print("  • Helm chart generation")
    print("  • Visual deployment pipeline builder")
    print("  • YAML generation from visual blocks")
    print("  • Complete CI/CD pipeline visualization")