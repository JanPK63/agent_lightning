#!/usr/bin/env python3
"""
Code RAG Service - Semantic code search with pgvector
Provides RAG capabilities for the Retriever Agent to search repositories
"""

import os
import sys
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import ast
import re

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import git

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeChunk(BaseModel):
    """Code chunk for indexing"""
    file_path: str = Field(description="File path")
    content: str = Field(description="Code content")
    language: str = Field(description="Programming language")
    function_name: Optional[str] = Field(default=None, description="Function/class name")
    start_line: int = Field(description="Start line number")
    end_line: int = Field(description="End line number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(description="Search query")
    language: Optional[str] = Field(default=None, description="Filter by language")
    limit: int = Field(default=10, description="Number of results")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")


class IndexRequest(BaseModel):
    """Repository indexing request"""
    repo_path: str = Field(description="Repository path")
    languages: List[str] = Field(default=["python", "javascript", "go", "java"], description="Languages to index")
    chunk_size: int = Field(default=50, description="Lines per chunk")


class CodeRAGService:
    """Code RAG Service for semantic code search"""
    
    def __init__(self):
        self.app = FastAPI(title="Code RAG Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("code_rag")
        self.cache = get_cache()
        
        # Initialize embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Database connection for pgvector
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agent_lightning"),
            "user": os.getenv("POSTGRES_USER", "agent_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agent_password")
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("✅ Code RAG Service initialized")
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _init_database(self):
        """Initialize pgvector database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Create pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create code embeddings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS code_embeddings (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    language VARCHAR(50),
                    function_name TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    embedding vector(%s),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    repo_hash VARCHAR(64),
                    UNIQUE(repo_hash, file_path, start_line)
                )
            """, (self.embedding_dim,))
            
            # Create index for similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS code_embeddings_embedding_idx 
                ON code_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            # Create repository index table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS indexed_repositories (
                    id SERIAL PRIMARY KEY,
                    repo_path TEXT UNIQUE NOT NULL,
                    repo_hash VARCHAR(64) UNIQUE NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_count INTEGER,
                    chunk_count INTEGER,
                    languages JSONB
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _extract_code_chunks(self, file_path: str, content: str, language: str, chunk_size: int = 50) -> List[CodeChunk]:
        """Extract code chunks from a file"""
        chunks = []
        lines = content.split('\n')
        
        # Language-specific parsing
        if language == "python":
            chunks.extend(self._extract_python_chunks(file_path, content, lines))
        elif language in ["javascript", "typescript"]:
            chunks.extend(self._extract_js_chunks(file_path, content, lines))
        elif language == "go":
            chunks.extend(self._extract_go_chunks(file_path, content, lines))
        elif language == "java":
            chunks.extend(self._extract_java_chunks(file_path, content, lines))
        
        # Also create sliding window chunks for general content
        for i in range(0, len(lines), chunk_size // 2):
            end = min(i + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[i:end])
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    language=language,
                    start_line=i + 1,
                    end_line=end,
                    metadata={"type": "window"}
                ))
        
        return chunks
    
    def _extract_python_chunks(self, file_path: str, content: str, lines: List[str]) -> List[CodeChunk]:
        """Extract Python functions and classes"""
        chunks = []
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    chunk_content = '\n'.join(lines[start_line-1:end_line])
                    
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        content=chunk_content,
                        language="python",
                        function_name=node.name,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={
                            "type": "class" if isinstance(node, ast.ClassDef) else "function",
                            "async": isinstance(node, ast.AsyncFunctionDef)
                        }
                    ))
        except:
            pass  # Fallback to window chunks if parsing fails
        
        return chunks
    
    def _extract_js_chunks(self, file_path: str, content: str, lines: List[str]) -> List[CodeChunk]:
        """Extract JavaScript/TypeScript functions and classes"""
        chunks = []
        
        # Simple regex-based extraction
        function_pattern = r'(async\s+)?function\s+(\w+)|const\s+(\w+)\s*=\s*(async\s*)?\([^)]*\)\s*=>|class\s+(\w+)'
        
        for match in re.finditer(function_pattern, content):
            name = match.group(2) or match.group(3) or match.group(5)
            if name:
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find the end of the function/class (simplified)
                brace_count = 0
                found_start = False
                end_line = start_line
                
                for i in range(start_line - 1, len(lines)):
                    line = lines[i]
                    for char in line:
                        if char == '{':
                            brace_count += 1
                            found_start = True
                        elif char == '}':
                            brace_count -= 1
                    
                    if found_start and brace_count == 0:
                        end_line = i + 1
                        break
                
                chunk_content = '\n'.join(lines[start_line-1:end_line])
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    language="javascript",
                    function_name=name,
                    start_line=start_line,
                    end_line=end_line,
                    metadata={"type": "function" if "class" not in match.group() else "class"}
                ))
        
        return chunks
    
    def _extract_go_chunks(self, file_path: str, content: str, lines: List[str]) -> List[CodeChunk]:
        """Extract Go functions and types"""
        chunks = []
        
        # Simple regex-based extraction for Go
        func_pattern = r'func\s+(\(.*?\)\s+)?(\w+)\s*\('
        type_pattern = r'type\s+(\w+)\s+(struct|interface)'
        
        for match in re.finditer(func_pattern, content):
            name = match.group(2)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            # Find the end of the function
            brace_count = 0
            found_start = False
            end_line = start_line
            
            for i in range(start_line - 1, len(lines)):
                line = lines[i]
                for char in line:
                    if char == '{':
                        brace_count += 1
                        found_start = True
                    elif char == '}':
                        brace_count -= 1
                
                if found_start and brace_count == 0:
                    end_line = i + 1
                    break
            
            chunk_content = '\n'.join(lines[start_line-1:end_line])
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                language="go",
                function_name=name,
                start_line=start_line,
                end_line=end_line,
                metadata={"type": "function"}
            ))
        
        return chunks
    
    def _extract_java_chunks(self, file_path: str, content: str, lines: List[str]) -> List[CodeChunk]:
        """Extract Java methods and classes"""
        chunks = []
        
        # Simple regex-based extraction for Java
        class_pattern = r'(public|private|protected)?\s*(abstract|final)?\s*class\s+(\w+)'
        method_pattern = r'(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\([^)]*\)'
        
        for match in re.finditer(class_pattern, content):
            name = match.group(3)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            # Find the end of the class
            brace_count = 0
            found_start = False
            end_line = start_line
            
            for i in range(start_line - 1, len(lines)):
                line = lines[i]
                for char in line:
                    if char == '{':
                        brace_count += 1
                        found_start = True
                    elif char == '}':
                        brace_count -= 1
                
                if found_start and brace_count == 0:
                    end_line = i + 1
                    break
            
            chunk_content = '\n'.join(lines[start_line-1:end_line])
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                language="java",
                function_name=name,
                start_line=start_line,
                end_line=end_line,
                metadata={"type": "class"}
            ))
        
        return chunks
    
    def _get_language_from_extension(self, file_path: str) -> Optional[str]:
        """Determine language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.rs': 'rust',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext)
    
    async def index_repository(self, repo_path: str, languages: List[str], chunk_size: int = 50):
        """Index a repository for RAG search"""
        try:
            # Calculate repo hash
            repo_hash = hashlib.sha256(repo_path.encode()).hexdigest()[:16]
            
            # Check if already indexed
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("SELECT id FROM indexed_repositories WHERE repo_hash = %s", (repo_hash,))
            if cur.fetchone():
                logger.info(f"Repository already indexed: {repo_path}")
                conn.close()
                return {"status": "already_indexed", "repo_hash": repo_hash}
            
            # Walk through repository
            file_count = 0
            chunk_count = 0
            indexed_languages = set()
            
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    # Determine language
                    language = self._get_language_from_extension(file_path)
                    if not language or language not in languages:
                        continue
                    
                    indexed_languages.add(language)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Extract chunks
                        chunks = self._extract_code_chunks(relative_path, content, language, chunk_size)
                        
                        # Generate embeddings and store
                        for chunk in chunks:
                            # Generate embedding
                            embedding = self.encoder.encode(chunk.content).tolist()
                            
                            # Store in database
                            cur.execute("""
                                INSERT INTO code_embeddings 
                                (file_path, content, language, function_name, start_line, end_line, embedding, metadata, repo_hash)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (repo_hash, file_path, start_line) DO NOTHING
                            """, (
                                chunk.file_path,
                                chunk.content,
                                chunk.language,
                                chunk.function_name,
                                chunk.start_line,
                                chunk.end_line,
                                embedding,
                                json.dumps(chunk.metadata),
                                repo_hash
                            ))
                            
                            chunk_count += 1
                        
                        file_count += 1
                        
                        if file_count % 10 == 0:
                            conn.commit()
                            logger.info(f"Indexed {file_count} files, {chunk_count} chunks")
                    
                    except Exception as e:
                        logger.error(f"Error indexing file {file_path}: {e}")
                        continue
            
            # Record indexed repository
            cur.execute("""
                INSERT INTO indexed_repositories (repo_path, repo_hash, file_count, chunk_count, languages)
                VALUES (%s, %s, %s, %s, %s)
            """, (repo_path, repo_hash, file_count, chunk_count, json.dumps(list(indexed_languages))))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Indexed repository: {file_count} files, {chunk_count} chunks")
            
            return {
                "status": "indexed",
                "repo_hash": repo_hash,
                "file_count": file_count,
                "chunk_count": chunk_count,
                "languages": list(indexed_languages)
            }
            
        except Exception as e:
            logger.error(f"Failed to index repository: {e}")
            raise
    
    async def search_code(self, query: str, language: Optional[str] = None, 
                         limit: int = 10, similarity_threshold: float = 0.7) -> List[Dict]:
        """Search for similar code chunks"""
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode(query).tolist()
            
            conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            cur = conn.cursor()
            
            # Build query
            if language:
                cur.execute("""
                    SELECT 
                        file_path, content, language, function_name, 
                        start_line, end_line, metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM code_embeddings
                    WHERE language = %s
                        AND 1 - (embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (query_embedding, language, query_embedding, similarity_threshold, limit))
            else:
                cur.execute("""
                    SELECT 
                        file_path, content, language, function_name, 
                        start_line, end_line, metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM code_embeddings
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (query_embedding, query_embedding, similarity_threshold, limit))
            
            results = cur.fetchall()
            conn.close()
            
            # Convert to list of dicts
            return [dict(r) for r in results]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "code_rag",
                "status": "healthy",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dim": self.embedding_dim,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/index")
        async def index_repository(request: IndexRequest, background_tasks: BackgroundTasks):
            """Index a repository"""
            background_tasks.add_task(
                self.index_repository,
                request.repo_path,
                request.languages,
                request.chunk_size
            )
            
            return {
                "status": "indexing_started",
                "repo_path": request.repo_path,
                "languages": request.languages
            }
        
        @self.app.post("/search")
        async def search(query: SearchQuery):
            """Search for similar code"""
            results = await self.search_code(
                query.query,
                query.language,
                query.limit,
                query.similarity_threshold
            )
            
            return {
                "query": query.query,
                "results": results,
                "count": len(results)
            }
        
        @self.app.get("/repositories")
        async def list_indexed_repositories():
            """List indexed repositories"""
            try:
                conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT repo_path, repo_hash, indexed_at, file_count, chunk_count, languages
                    FROM indexed_repositories
                    ORDER BY indexed_at DESC
                """)
                
                repos = cur.fetchall()
                conn.close()
                
                return {
                    "repositories": [dict(r) for r in repos],
                    "count": len(repos)
                }
            except Exception as e:
                logger.error(f"Failed to list repositories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/repositories/{repo_hash}")
        async def delete_repository(repo_hash: str):
            """Delete an indexed repository"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                
                # Delete embeddings
                cur.execute("DELETE FROM code_embeddings WHERE repo_hash = %s", (repo_hash,))
                deleted_chunks = cur.rowcount
                
                # Delete repository record
                cur.execute("DELETE FROM indexed_repositories WHERE repo_hash = %s", (repo_hash,))
                
                conn.commit()
                conn.close()
                
                return {
                    "status": "deleted",
                    "repo_hash": repo_hash,
                    "deleted_chunks": deleted_chunks
                }
            except Exception as e:
                logger.error(f"Failed to delete repository: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_statistics():
            """Get indexing statistics"""
            try:
                conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
                cur = conn.cursor()
                
                # Get counts
                cur.execute("SELECT COUNT(*) as total_chunks FROM code_embeddings")
                total_chunks = cur.fetchone()['total_chunks']
                
                cur.execute("SELECT COUNT(*) as total_repos FROM indexed_repositories")
                total_repos = cur.fetchone()['total_repos']
                
                # Get language distribution
                cur.execute("""
                    SELECT language, COUNT(*) as count 
                    FROM code_embeddings 
                    GROUP BY language 
                    ORDER BY count DESC
                """)
                language_dist = cur.fetchall()
                
                conn.close()
                
                return {
                    "total_repositories": total_repos,
                    "total_chunks": total_chunks,
                    "language_distribution": [dict(r) for r in language_dist]
                }
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Code RAG Service starting up...")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Code RAG Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = CodeRAGService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("CODE_RAG_PORT", 8017))
    logger.info(f"Starting Code RAG Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()