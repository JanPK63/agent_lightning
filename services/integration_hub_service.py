#!/usr/bin/env python3
"""
Integration Hub Microservice
Handles all external integrations: Salesforce, Slack, databases, APIs, webhooks
Based on the architecture from technical_architecture.md
"""

import os
import sys
import json
import asyncio
import aiohttp
import uuid
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging
import base64
from urllib.parse import urlparse, urlencode

from fastapi import FastAPI, HTTPException, Depends, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectorType(str, Enum):
    """Types of integration connectors"""
    SALESFORCE = "salesforce"
    SLACK = "slack"
    GITHUB = "github"
    JIRA = "jira"
    GOOGLE_WORKSPACE = "google_workspace"
    MICROSOFT_365 = "microsoft_365"
    AWS = "aws"
    AZURE = "azure"
    DATABASE = "database"
    REST_API = "rest_api"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"
    CUSTOM = "custom"


class AuthType(str, Enum):
    """Authentication types"""
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BASIC = "basic"
    JWT = "jwt"
    SAML = "saml"
    CUSTOM = "custom"
    NONE = "none"


class IntegrationStatus(str, Enum):
    """Integration connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"
    RATE_LIMITED = "rate_limited"


class DataFormat(str, Enum):
    """Data transformation formats"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    FORM_DATA = "form_data"
    BINARY = "binary"


# Pydantic Models
class AuthConfig(BaseModel):
    """Authentication configuration"""
    auth_type: AuthType
    credentials: Dict[str, Any] = Field(default_factory=dict)
    oauth_config: Optional[Dict[str, Any]] = None
    token_endpoint: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None


class ConnectorConfig(BaseModel):
    """Connector configuration"""
    connector_type: ConnectorType
    name: str
    description: Optional[str] = None
    base_url: Optional[str] = None
    auth_config: AuthConfig
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: int = 30
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 3, "delay": 1})
    rate_limit: Optional[Dict[str, Any]] = None
    custom_config: Dict[str, Any] = Field(default_factory=dict)


class Integration(BaseModel):
    """Integration instance"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str
    name: str
    connector_type: ConnectorType
    config: ConnectorConfig
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataTransformation(BaseModel):
    """Data transformation configuration"""
    input_format: DataFormat
    output_format: DataFormat
    mapping: Dict[str, Any] = Field(default_factory=dict)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)


class IntegrationRequest(BaseModel):
    """Request to execute an integration"""
    integration_id: str
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    transform: Optional[DataTransformation] = None
    async_execution: bool = False
    callback_url: Optional[HttpUrl] = None


class CreateIntegrationRequest(BaseModel):
    """Request to create a new integration"""
    name: str
    connector_type: ConnectorType
    config: ConnectorConfig
    test_connection: bool = True


class WebhookRegistration(BaseModel):
    """Webhook registration"""
    integration_id: str
    endpoint_path: str
    events: List[str]
    secret: Optional[str] = None
    active: bool = True


class IntegrationResult(BaseModel):
    """Result of integration execution"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    duration_ms: Optional[int] = None


class BaseConnector:
    """Base class for all connectors"""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.rate_limit_remaining: int = 1000
        self.rate_limit_reset: Optional[datetime] = None
    
    async def connect(self) -> bool:
        """Establish connection"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()
    
    async def execute(self, action: str, payload: Dict[str, Any]) -> IntegrationResult:
        """Execute integration action"""
        raise NotImplementedError
    
    async def test_connection(self) -> bool:
        """Test if connection is working"""
        raise NotImplementedError
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Tuple[int, Any, Dict[str, str]]:
        """Make HTTP request with auth and retry logic"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Add authentication
        headers = kwargs.get('headers', {})
        headers.update(self.config.headers)
        
        if self.config.auth_config.auth_type == AuthType.API_KEY:
            headers['Authorization'] = f"Bearer {self.config.auth_config.credentials.get('api_key')}"
        elif self.config.auth_config.auth_type == AuthType.BASIC:
            username = self.config.auth_config.credentials.get('username')
            password = self.config.auth_config.credentials.get('password')
            auth_str = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers['Authorization'] = f"Basic {auth_str}"
        
        kwargs['headers'] = headers
        
        # Retry logic
        max_attempts = self.config.retry_policy.get('max_attempts', 3)
        delay = self.config.retry_policy.get('delay', 1)
        
        for attempt in range(max_attempts):
            try:
                async with self.session.request(method, url, timeout=self.config.timeout, **kwargs) as response:
                    # Check rate limiting
                    if 'X-RateLimit-Remaining' in response.headers:
                        self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                    
                    # Parse response
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                    else:
                        data = await response.text()
                    
                    return response.status, data, dict(response.headers)
                    
            except asyncio.TimeoutError:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(delay)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(delay)


class SalesforceConnector(BaseConnector):
    """Salesforce integration connector"""
    
    async def connect(self) -> bool:
        """Connect to Salesforce"""
        try:
            # OAuth2 flow for Salesforce
            token_url = "https://login.salesforce.com/services/oauth2/token"
            
            data = {
                'grant_type': 'password',
                'client_id': self.config.auth_config.credentials.get('client_id'),
                'client_secret': self.config.auth_config.credentials.get('client_secret'),
                'username': self.config.auth_config.credentials.get('username'),
                'password': self.config.auth_config.credentials.get('password')
            }
            
            # In production, would make actual OAuth request
            # For now, simulate successful connection
            self.auth_token = "simulated_salesforce_token"
            return True
            
        except Exception as e:
            logger.error(f"Salesforce connection error: {e}")
            return False
    
    async def execute(self, action: str, payload: Dict[str, Any]) -> IntegrationResult:
        """Execute Salesforce action"""
        try:
            if action == "create_lead":
                # Simulate creating a lead
                return IntegrationResult(
                    success=True,
                    data={
                        "id": str(uuid.uuid4()),
                        "status": "created",
                        "lead": payload
                    }
                )
            
            elif action == "query":
                # Simulate SOQL query
                return IntegrationResult(
                    success=True,
                    data={
                        "records": [
                            {"Id": "001XX000003DHPh", "Name": "Sample Account"}
                        ],
                        "totalSize": 1
                    }
                )
            
            elif action == "update_opportunity":
                # Simulate updating opportunity
                return IntegrationResult(
                    success=True,
                    data={"id": payload.get("id"), "updated": True}
                )
            
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def test_connection(self) -> bool:
        """Test Salesforce connection"""
        return await self.connect()


class SlackConnector(BaseConnector):
    """Slack integration connector"""
    
    async def connect(self) -> bool:
        """Connect to Slack"""
        try:
            self.auth_token = self.config.auth_config.credentials.get('bot_token')
            return True
        except Exception as e:
            logger.error(f"Slack connection error: {e}")
            return False
    
    async def execute(self, action: str, payload: Dict[str, Any]) -> IntegrationResult:
        """Execute Slack action"""
        try:
            if action == "send_message":
                # Simulate sending message
                return IntegrationResult(
                    success=True,
                    data={
                        "ok": True,
                        "channel": payload.get("channel"),
                        "ts": datetime.now().timestamp(),
                        "message": payload.get("text")
                    }
                )
            
            elif action == "create_channel":
                # Simulate creating channel
                return IntegrationResult(
                    success=True,
                    data={
                        "ok": True,
                        "channel": {
                            "id": f"C{uuid.uuid4().hex[:10].upper()}",
                            "name": payload.get("name")
                        }
                    }
                )
            
            elif action == "upload_file":
                # Simulate file upload
                return IntegrationResult(
                    success=True,
                    data={
                        "ok": True,
                        "file": {
                            "id": f"F{uuid.uuid4().hex[:10].upper()}",
                            "name": payload.get("filename")
                        }
                    }
                )
            
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def test_connection(self) -> bool:
        """Test Slack connection"""
        return self.auth_token is not None


class DatabaseConnector(BaseConnector):
    """Database integration connector"""
    
    async def connect(self) -> bool:
        """Connect to database"""
        try:
            # In production, would establish actual database connection
            # Support for PostgreSQL, MySQL, MongoDB, etc.
            db_type = self.config.custom_config.get('db_type', 'postgresql')
            logger.info(f"Connected to {db_type} database")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    async def execute(self, action: str, payload: Dict[str, Any]) -> IntegrationResult:
        """Execute database action"""
        try:
            if action == "query":
                # Simulate query execution
                return IntegrationResult(
                    success=True,
                    data={
                        "rows": [
                            {"id": 1, "name": "Sample Data"}
                        ],
                        "row_count": 1
                    }
                )
            
            elif action == "insert":
                # Simulate insert
                return IntegrationResult(
                    success=True,
                    data={"inserted_id": str(uuid.uuid4()), "rows_affected": 1}
                )
            
            elif action == "update":
                # Simulate update
                return IntegrationResult(
                    success=True,
                    data={"rows_affected": payload.get("count", 1)}
                )
            
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        return await self.connect()


class RESTAPIConnector(BaseConnector):
    """Generic REST API connector"""
    
    async def connect(self) -> bool:
        """Initialize REST API connection"""
        return True
    
    async def execute(self, action: str, payload: Dict[str, Any]) -> IntegrationResult:
        """Execute REST API call"""
        try:
            method = payload.get('method', 'GET').upper()
            url = f"{self.config.base_url}{payload.get('endpoint', '')}"
            
            # Build request
            kwargs = {}
            if method in ['POST', 'PUT', 'PATCH']:
                kwargs['json'] = payload.get('body', {})
            
            if payload.get('params'):
                kwargs['params'] = payload['params']
            
            # Make request
            status, data, headers = await self._make_request(method, url, **kwargs)
            
            return IntegrationResult(
                success=status < 400,
                data=data,
                status_code=status,
                headers=headers
            )
            
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def test_connection(self) -> bool:
        """Test REST API connection"""
        if not self.config.base_url:
            return False
        
        try:
            # Try to access base URL
            status, _, _ = await self._make_request('GET', self.config.base_url)
            return status < 500
        except:
            return False


class IntegrationHubService:
    """Main Integration Hub Service"""
    
    def __init__(self):
        self.app = FastAPI(title="Integration Hub Service", version="1.0.0")
        
        # Storage (in-memory for now)
        self.integrations: Dict[str, Integration] = {}
        self.connectors: Dict[str, BaseConnector] = {}
        self.webhooks: Dict[str, WebhookRegistration] = {}
        
        # Connector registry
        self.connector_classes = {
            ConnectorType.SALESFORCE: SalesforceConnector,
            ConnectorType.SLACK: SlackConnector,
            ConnectorType.DATABASE: DatabaseConnector,
            ConnectorType.REST_API: RESTAPIConnector,
            # Add more connectors as needed
        }
        
        # Metrics
        self.metrics = {
            "total_integrations": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_data_processed": 0
        }
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        self._load_sample_integrations()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_event_handlers(self):
        """Setup startup and shutdown handlers"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize service"""
            logger.info("Integration Hub Service started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            # Disconnect all connectors
            for connector in self.connectors.values():
                await connector.disconnect()
            logger.info("Integration Hub Service shut down")
    
    def _load_sample_integrations(self):
        """Load sample integrations"""
        # Salesforce integration
        salesforce = Integration(
            id="int-salesforce-001",
            organization_id="default-org",
            name="Salesforce CRM",
            connector_type=ConnectorType.SALESFORCE,
            config=ConnectorConfig(
                connector_type=ConnectorType.SALESFORCE,
                name="Salesforce Production",
                base_url="https://mycompany.salesforce.com",
                auth_config=AuthConfig(
                    auth_type=AuthType.OAUTH2,
                    credentials={
                        "client_id": "sample_client_id",
                        "client_secret": "sample_secret"
                    }
                )
            ),
            status=IntegrationStatus.CONNECTED
        )
        self.integrations[salesforce.id] = salesforce
        
        # Slack integration
        slack = Integration(
            id="int-slack-001",
            organization_id="default-org",
            name="Slack Workspace",
            connector_type=ConnectorType.SLACK,
            config=ConnectorConfig(
                connector_type=ConnectorType.SLACK,
                name="Company Slack",
                base_url="https://slack.com/api",
                auth_config=AuthConfig(
                    auth_type=AuthType.API_KEY,
                    credentials={"bot_token": "xoxb-sample-token"}
                )
            ),
            status=IntegrationStatus.CONNECTED
        )
        self.integrations[slack.id] = slack
        
        logger.info(f"Loaded {len(self.integrations)} sample integrations")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check"""
            return {
                "status": "healthy",
                "service": "integration_hub",
                "timestamp": datetime.now().isoformat(),
                "active_integrations": len([i for i in self.integrations.values() 
                                          if i.status == IntegrationStatus.CONNECTED])
            }
        
        # Integration management
        @self.app.post("/api/v1/integrations", response_model=Integration)
        async def create_integration(request: CreateIntegrationRequest, req: Request):
            """Create a new integration"""
            org_id = req.headers.get("X-Organization-ID", "default-org")
            
            integration = Integration(
                organization_id=org_id,
                name=request.name,
                connector_type=request.connector_type,
                config=request.config
            )
            
            # Test connection if requested
            if request.test_connection:
                connector = await self._get_or_create_connector(integration)
                if await connector.test_connection():
                    integration.status = IntegrationStatus.CONNECTED
                else:
                    integration.status = IntegrationStatus.ERROR
            
            self.integrations[integration.id] = integration
            self.metrics["total_integrations"] += 1
            
            logger.info(f"Created integration: {integration.id}")
            return integration
        
        @self.app.get("/api/v1/integrations", response_model=List[Integration])
        async def list_integrations(req: Request):
            """List all integrations"""
            org_id = req.headers.get("X-Organization-ID", "default-org")
            return [i for i in self.integrations.values() if i.organization_id == org_id]
        
        @self.app.get("/api/v1/integrations/{integration_id}", response_model=Integration)
        async def get_integration(integration_id: str):
            """Get integration details"""
            if integration_id not in self.integrations:
                raise HTTPException(status_code=404, detail="Integration not found")
            return self.integrations[integration_id]
        
        @self.app.delete("/api/v1/integrations/{integration_id}")
        async def delete_integration(integration_id: str):
            """Delete an integration"""
            if integration_id not in self.integrations:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            # Disconnect if connected
            if integration_id in self.connectors:
                await self.connectors[integration_id].disconnect()
                del self.connectors[integration_id]
            
            del self.integrations[integration_id]
            logger.info(f"Deleted integration: {integration_id}")
            return {"message": "Integration deleted successfully"}
        
        # Integration execution
        @self.app.post("/api/v1/integrations/execute", response_model=IntegrationResult)
        async def execute_integration(request: IntegrationRequest, background_tasks: BackgroundTasks):
            """Execute an integration action"""
            if request.integration_id not in self.integrations:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            integration = self.integrations[request.integration_id]
            
            if request.async_execution:
                # Queue for async execution
                background_tasks.add_task(self._execute_integration_async, 
                                        request, integration)
                return IntegrationResult(
                    success=True,
                    data={"status": "queued", "integration_id": request.integration_id}
                )
            else:
                # Execute synchronously
                return await self._execute_integration(request, integration)
        
        @self.app.post("/api/v1/integrations/{integration_id}/test")
        async def test_connection(integration_id: str):
            """Test integration connection"""
            if integration_id not in self.integrations:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            integration = self.integrations[integration_id]
            connector = await self._get_or_create_connector(integration)
            
            success = await connector.test_connection()
            integration.status = IntegrationStatus.CONNECTED if success else IntegrationStatus.ERROR
            integration.updated_at = datetime.now()
            
            return {"success": success, "status": integration.status}
        
        # Webhooks
        @self.app.post("/api/v1/webhooks/register", response_model=WebhookRegistration)
        async def register_webhook(webhook: WebhookRegistration):
            """Register a webhook"""
            webhook_id = str(uuid.uuid4())
            self.webhooks[webhook_id] = webhook
            logger.info(f"Registered webhook: {webhook_id}")
            return webhook
        
        @self.app.post("/api/v1/webhooks/{webhook_id}")
        async def receive_webhook(webhook_id: str, request: Request):
            """Receive webhook callback"""
            if webhook_id not in self.webhooks:
                raise HTTPException(status_code=404, detail="Webhook not found")
            
            webhook = self.webhooks[webhook_id]
            
            # Verify webhook signature if configured
            if webhook.secret:
                signature = request.headers.get("X-Signature")
                body = await request.body()
                expected = hmac.new(webhook.secret.encode(), body, hashlib.sha256).hexdigest()
                
                if signature != expected:
                    raise HTTPException(status_code=401, detail="Invalid signature")
            
            # Process webhook
            data = await request.json()
            logger.info(f"Received webhook: {webhook_id}")
            
            return {"status": "received", "webhook_id": webhook_id}
        
        # Connector types
        @self.app.get("/api/v1/connectors")
        async def list_connector_types():
            """List available connector types"""
            return [
                {
                    "type": connector_type.value,
                    "name": connector_type.value.replace("_", " ").title(),
                    "supported": connector_type in self.connector_classes
                }
                for connector_type in ConnectorType
            ]
        
        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            """Get service metrics"""
            return {
                "metrics": self.metrics,
                "integrations": {
                    "total": len(self.integrations),
                    "connected": len([i for i in self.integrations.values() 
                                    if i.status == IntegrationStatus.CONNECTED]),
                    "error": len([i for i in self.integrations.values() 
                                if i.status == IntegrationStatus.ERROR])
                },
                "webhooks": len(self.webhooks)
            }
    
    async def _get_or_create_connector(self, integration: Integration) -> BaseConnector:
        """Get or create a connector instance"""
        if integration.id not in self.connectors:
            connector_class = self.connector_classes.get(
                integration.connector_type,
                RESTAPIConnector  # Default to REST API
            )
            self.connectors[integration.id] = connector_class(integration.config)
        
        return self.connectors[integration.id]
    
    async def _execute_integration(self, request: IntegrationRequest, 
                                  integration: Integration) -> IntegrationResult:
        """Execute integration synchronously"""
        try:
            # Get connector
            connector = await self._get_or_create_connector(integration)
            
            # Ensure connected
            if integration.status != IntegrationStatus.CONNECTED:
                if not await connector.connect():
                    return IntegrationResult(
                        success=False,
                        error="Failed to connect to integration"
                    )
                integration.status = IntegrationStatus.CONNECTED
            
            # Transform input data if needed
            payload = request.payload
            if request.transform:
                payload = self._transform_data(payload, request.transform)
            
            # Execute action
            result = await connector.execute(request.action, payload)
            
            # Update metrics
            if result.success:
                self.metrics["successful_calls"] += 1
            else:
                self.metrics["failed_calls"] += 1
            
            # Update last used
            integration.last_used = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Integration execution error: {e}")
            self.metrics["failed_calls"] += 1
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def _execute_integration_async(self, request: IntegrationRequest, 
                                        integration: Integration):
        """Execute integration asynchronously"""
        result = await self._execute_integration(request, integration)
        
        # Send callback if configured
        if request.callback_url:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        str(request.callback_url),
                        json=asdict(result)
                    )
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _transform_data(self, data: Any, transformation: DataTransformation) -> Any:
        """Transform data between formats"""
        # Simple transformation logic (would be more complex in production)
        if transformation.mapping:
            transformed = {}
            for target_key, source_key in transformation.mapping.items():
                if isinstance(source_key, str) and source_key in data:
                    transformed[target_key] = data[source_key]
                else:
                    transformed[target_key] = source_key
            return transformed
        return data


def create_service():
    """Create and return the service instance"""
    return IntegrationHubService()


if __name__ == "__main__":
    import uvicorn
    
    print("Integration Hub Microservice")
    print("=" * 60)
    
    service = create_service()
    
    print("\n🔌 Starting Integration Hub Service on port 8004")
    print("\nEndpoints:")
    print("  • POST /api/v1/integrations - Create integration")
    print("  • GET  /api/v1/integrations - List integrations")
    print("  • POST /api/v1/integrations/execute - Execute integration")
    print("  • POST /api/v1/webhooks/register - Register webhook")
    print("  • GET  /api/v1/connectors - List connector types")
    
    print("\n🔗 Available Connectors:")
    print("  • Salesforce")
    print("  • Slack")
    print("  • Database (PostgreSQL, MySQL, MongoDB)")
    print("  • REST API (Generic)")
    print("  • More coming soon...")
    
    uvicorn.run(service.app, host="0.0.0.0", port=8004, reload=False)