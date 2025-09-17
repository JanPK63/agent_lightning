#!/usr/bin/env python3
"""
MCP Protocol Adapter for Agent Lightning
Wraps our memory and knowledge services with MCP protocol compliance
"""

import asyncio
import json
from typing import Dict, List, Any
import httpx
from fastapi import FastAPI
import uvicorn

class AgentLightningMCPServer:
    """MCP-compatible server for Agent Lightning services"""
    
    def __init__(self):
        self.app = FastAPI(title="Agent Lightning MCP Server")
        self.memory_url = "http://memory-manager:8012"
        self.knowledge_url = "http://knowledge-manager:8014"
        self.setup_routes()
    
    def setup_routes(self):
        """Setup MCP-compatible routes"""
        
        @self.app.get("/mcp/resources")
        async def list_resources():
            """List available MCP resources"""
            return {
                "resources": [
                    {
                        "uri": "memory://episodic",
                        "name": "Episodic Memory",
                        "description": "Agent episodic memory storage"
                    },
                    {
                        "uri": "memory://semantic", 
                        "name": "Semantic Memory",
                        "description": "Agent semantic knowledge"
                    },
                    {
                        "uri": "knowledge://base",
                        "name": "Knowledge Base",
                        "description": "Agent knowledge repository"
                    }
                ]
            }
        
        @self.app.get("/mcp/resource/{uri:path}")
        async def read_resource(uri: str):
            """Read MCP resource content"""
            if uri.startswith("memory/"):
                memory_type = uri.split("/")[1]
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.memory_url}/memory/retrieve",
                        json={
                            "agent_id": "default",
                            "memory_type": memory_type,
                            "limit": 50
                        }
                    )
                    return response.json()
            
            elif uri.startswith("knowledge/"):
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.knowledge_url}/agents/default/knowledge"
                    )
                    return response.json()
            
            return {"error": "Resource not found"}
        
        @self.app.get("/mcp/tools")
        async def list_tools():
            """List available MCP tools"""
            return {
                "tools": [
                    {
                        "name": "store_memory",
                        "description": "Store memory item",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string"},
                                "content": {"type": "object"},
                                "memory_type": {"type": "string"},
                                "importance": {"type": "number"}
                            }
                        }
                    },
                    {
                        "name": "search_knowledge",
                        "description": "Search knowledge base",
                        "inputSchema": {
                            "type": "object", 
                            "properties": {
                                "agent_name": {"type": "string"},
                                "query": {"type": "string"},
                                "limit": {"type": "number"}
                            }
                        }
                    }
                ]
            }
        
        @self.app.post("/mcp/tools/{tool_name}")
        async def call_tool(tool_name: str, arguments: Dict[str, Any]):
            """Execute MCP tool calls"""
            if tool_name == "store_memory":
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.memory_url}/memory/store",
                        json=arguments
                    )
                    return response.json()
            
            elif tool_name == "search_knowledge":
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.knowledge_url}/search",
                        json=arguments
                    )
                    return response.json()
            
            return {"error": "Unknown tool"}

if __name__ == "__main__":
    server = AgentLightningMCPServer()
    uvicorn.run(server.app, host="0.0.0.0", port=8015)