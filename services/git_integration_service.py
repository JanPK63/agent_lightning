#!/usr/bin/env python3
"""
Git/GitHub Integration Service
Handles Git operations and GitHub PR management for the Integrator Agent
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import git
from github import Github, GithubException
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitConfig(BaseModel):
    """Git configuration"""
    repo_path: str = Field(description="Repository path")
    remote_url: Optional[str] = Field(default=None, description="Remote repository URL")
    branch: Optional[str] = Field(default="main", description="Branch name")
    author_name: Optional[str] = Field(default="AI Agent", description="Commit author name")
    author_email: Optional[str] = Field(default="ai@agent.local", description="Commit author email")


class CommitRequest(BaseModel):
    """Git commit request"""
    repo_path: str = Field(description="Repository path")
    message: str = Field(description="Commit message")
    files: Optional[List[str]] = Field(default=None, description="Specific files to commit")
    branch: Optional[str] = Field(default=None, description="Branch to commit to")


class PRRequest(BaseModel):
    """Pull request creation request"""
    repo_path: str = Field(description="Repository path")
    title: str = Field(description="PR title")
    body: str = Field(description="PR description")
    base_branch: str = Field(default="main", description="Base branch")
    head_branch: str = Field(description="Head branch")
    draft: bool = Field(default=False, description="Create as draft PR")
    labels: Optional[List[str]] = Field(default=None, description="PR labels")
    assignees: Optional[List[str]] = Field(default=None, description="PR assignees")
    reviewers: Optional[List[str]] = Field(default=None, description="PR reviewers")


class BranchRequest(BaseModel):
    """Branch operation request"""
    repo_path: str = Field(description="Repository path")
    branch_name: str = Field(description="Branch name")
    from_branch: Optional[str] = Field(default="main", description="Source branch")


class MergeRequest(BaseModel):
    """Merge request"""
    repo_path: str = Field(description="Repository path")
    source_branch: str = Field(description="Source branch")
    target_branch: str = Field(description="Target branch")
    commit_message: Optional[str] = Field(default=None, description="Merge commit message")
    squash: bool = Field(default=False, description="Squash commits")


class ReviewComment(BaseModel):
    """PR review comment"""
    pr_number: int = Field(description="PR number")
    body: str = Field(description="Comment body")
    path: Optional[str] = Field(default=None, description="File path for inline comment")
    line: Optional[int] = Field(default=None, description="Line number for inline comment")
    side: Optional[str] = Field(default="RIGHT", description="Side of diff (LEFT/RIGHT)")


class GitIntegrationService:
    """Git/GitHub Integration Service"""
    
    def __init__(self):
        self.app = FastAPI(title="Git Integration Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("git_integration")
        self.cache = get_cache()
        
        # GitHub client (initialized with token if available)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_client = Github(self.github_token) if self.github_token else None
        
        # Git operations cache
        self.repo_cache = {}
        
        logger.info("✅ Git Integration Service initialized")
        
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
    
    def _get_repo(self, repo_path: str) -> git.Repo:
        """Get or create git repo object"""
        if repo_path not in self.repo_cache:
            try:
                self.repo_cache[repo_path] = git.Repo(repo_path)
            except git.InvalidGitRepositoryError:
                # Initialize new repo if not exists
                self.repo_cache[repo_path] = git.Repo.init(repo_path)
        return self.repo_cache[repo_path]
    
    async def create_branch(self, repo_path: str, branch_name: str, 
                           from_branch: str = "main") -> Dict[str, Any]:
        """Create a new branch"""
        try:
            repo = self._get_repo(repo_path)
            
            # Checkout source branch first
            if from_branch in repo.heads:
                repo.heads[from_branch].checkout()
            else:
                # Try to fetch from remote
                origin = repo.remote("origin")
                origin.fetch()
                repo.create_head(from_branch, origin.refs[from_branch])
                repo.heads[from_branch].checkout()
            
            # Create new branch
            new_branch = repo.create_head(branch_name)
            new_branch.checkout()
            
            logger.info(f"Created branch {branch_name} from {from_branch}")
            
            return {
                "status": "success",
                "branch": branch_name,
                "from_branch": from_branch,
                "commit": str(repo.head.commit)
            }
            
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def commit_changes(self, repo_path: str, message: str, 
                            files: Optional[List[str]] = None,
                            branch: Optional[str] = None) -> Dict[str, Any]:
        """Commit changes to repository"""
        try:
            repo = self._get_repo(repo_path)
            
            # Checkout branch if specified
            if branch and branch in repo.heads:
                repo.heads[branch].checkout()
            
            # Stage files
            if files:
                repo.index.add(files)
            else:
                # Stage all changes
                repo.git.add(A=True)
            
            # Check if there are changes to commit
            if not repo.index.diff("HEAD") and not repo.untracked_files:
                return {
                    "status": "no_changes",
                    "message": "No changes to commit"
                }
            
            # Commit
            commit = repo.index.commit(message)
            
            logger.info(f"Created commit {commit.hexsha[:8]}: {message}")
            
            # Record in database
            commit_data = {
                "repo_path": repo_path,
                "commit_sha": commit.hexsha,
                "message": message,
                "branch": repo.active_branch.name,
                "author": str(commit.author),
                "timestamp": datetime.fromtimestamp(commit.committed_date).isoformat()
            }
            
            # Store in cache
            cache_key = f"commit:{commit.hexsha[:8]}"
            self.cache.set(cache_key, commit_data, ttl=3600)
            
            return {
                "status": "success",
                "commit_sha": commit.hexsha,
                "message": message,
                "branch": repo.active_branch.name,
                "files_changed": len(commit.stats.files),
                "stats": {
                    "insertions": commit.stats.total["insertions"],
                    "deletions": commit.stats.total["deletions"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def push_branch(self, repo_path: str, branch: str, 
                         remote: str = "origin") -> Dict[str, Any]:
        """Push branch to remote"""
        try:
            repo = self._get_repo(repo_path)
            
            # Get remote
            origin = repo.remote(remote)
            
            # Push branch
            push_info = origin.push(branch)[0]
            
            logger.info(f"Pushed branch {branch} to {remote}")
            
            return {
                "status": "success",
                "branch": branch,
                "remote": remote,
                "summary": push_info.summary
            }
            
        except Exception as e:
            logger.error(f"Failed to push branch: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_pull_request(self, request: PRRequest) -> Dict[str, Any]:
        """Create GitHub pull request"""
        if not self.github_client:
            raise HTTPException(status_code=401, detail="GitHub token not configured")
        
        try:
            repo = self._get_repo(request.repo_path)
            
            # Get GitHub repository
            remote_url = repo.remote("origin").url
            
            # Parse owner and repo name from URL
            # Handle both SSH and HTTPS URLs
            if remote_url.startswith("git@"):
                # SSH URL: git@github.com:owner/repo.git
                parts = remote_url.split(":")[-1].replace(".git", "").split("/")
            else:
                # HTTPS URL: https://github.com/owner/repo.git
                parts = remote_url.replace(".git", "").split("/")[-2:]
            
            owner, repo_name = parts[0], parts[1]
            
            # Get GitHub repo
            github_repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            
            # Create PR
            pr = github_repo.create_pull(
                title=request.title,
                body=request.body,
                base=request.base_branch,
                head=request.head_branch,
                draft=request.draft
            )
            
            # Add labels if specified
            if request.labels:
                pr.add_to_labels(*request.labels)
            
            # Add assignees if specified
            if request.assignees:
                pr.add_to_assignees(*request.assignees)
            
            # Request reviewers if specified
            if request.reviewers:
                pr.create_review_request(reviewers=request.reviewers)
            
            logger.info(f"Created PR #{pr.number}: {pr.title}")
            
            # Store PR info
            pr_data = {
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "url": pr.html_url,
                "created_at": pr.created_at.isoformat(),
                "repo": f"{owner}/{repo_name}"
            }
            
            # Cache PR data
            cache_key = f"pr:{owner}/{repo_name}/{pr.number}"
            self.cache.set(cache_key, pr_data, ttl=3600)
            
            return pr_data
            
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            raise HTTPException(status_code=e.status, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to create PR: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def add_review_comment(self, comment: ReviewComment) -> Dict[str, Any]:
        """Add review comment to PR"""
        if not self.github_client:
            raise HTTPException(status_code=401, detail="GitHub token not configured")
        
        try:
            # Get PR from cache or API
            cache_key = f"pr:*/*/{comment.pr_number}"
            pr_data = self.cache.get(cache_key)
            
            if not pr_data:
                raise HTTPException(status_code=404, detail="PR not found")
            
            # Get GitHub repo and PR
            github_repo = self.github_client.get_repo(pr_data["repo"])
            pr = github_repo.get_pull(comment.pr_number)
            
            # Add comment
            if comment.path and comment.line:
                # Inline comment
                pr_comment = pr.create_review_comment(
                    body=comment.body,
                    path=comment.path,
                    line=comment.line,
                    side=comment.side
                )
            else:
                # General PR comment
                pr_comment = pr.create_issue_comment(comment.body)
            
            logger.info(f"Added comment to PR #{comment.pr_number}")
            
            return {
                "status": "success",
                "pr_number": comment.pr_number,
                "comment_id": pr_comment.id,
                "created_at": pr_comment.created_at.isoformat()
            }
            
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            raise HTTPException(status_code=e.status, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to add comment: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def merge_pull_request(self, pr_number: int, repo_path: str,
                                 commit_message: Optional[str] = None,
                                 merge_method: str = "merge") -> Dict[str, Any]:
        """Merge a pull request"""
        if not self.github_client:
            raise HTTPException(status_code=401, detail="GitHub token not configured")
        
        try:
            repo = self._get_repo(repo_path)
            remote_url = repo.remote("origin").url
            
            # Parse owner and repo name
            if remote_url.startswith("git@"):
                parts = remote_url.split(":")[-1].replace(".git", "").split("/")
            else:
                parts = remote_url.replace(".git", "").split("/")[-2:]
            
            owner, repo_name = parts[0], parts[1]
            
            # Get GitHub repo and PR
            github_repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            pr = github_repo.get_pull(pr_number)
            
            # Check if PR is mergeable
            if not pr.mergeable:
                return {
                    "status": "not_mergeable",
                    "pr_number": pr_number,
                    "message": "PR has conflicts or is not mergeable"
                }
            
            # Merge PR
            if merge_method == "squash":
                result = pr.merge(commit_message=commit_message, merge_method="squash")
            elif merge_method == "rebase":
                result = pr.merge(commit_message=commit_message, merge_method="rebase")
            else:
                result = pr.merge(commit_message=commit_message)
            
            logger.info(f"Merged PR #{pr_number}")
            
            return {
                "status": "success",
                "pr_number": pr_number,
                "merged": result.merged,
                "message": result.message,
                "sha": result.sha
            }
            
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            raise HTTPException(status_code=e.status, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to merge PR: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_pr_status(self, pr_number: int, repo_path: str) -> Dict[str, Any]:
        """Get PR status and checks"""
        if not self.github_client:
            raise HTTPException(status_code=401, detail="GitHub token not configured")
        
        try:
            repo = self._get_repo(repo_path)
            remote_url = repo.remote("origin").url
            
            # Parse owner and repo name
            if remote_url.startswith("git@"):
                parts = remote_url.split(":")[-1].replace(".git", "").split("/")
            else:
                parts = remote_url.replace(".git", "").split("/")[-2:]
            
            owner, repo_name = parts[0], parts[1]
            
            # Get GitHub repo and PR
            github_repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            pr = github_repo.get_pull(pr_number)
            
            # Get commit status
            last_commit = pr.get_commits().reversed[0]
            statuses = last_commit.get_statuses()
            
            # Get check runs
            check_runs = last_commit.get_check_runs()
            
            return {
                "pr_number": pr_number,
                "state": pr.state,
                "mergeable": pr.mergeable,
                "mergeable_state": pr.mergeable_state,
                "title": pr.title,
                "url": pr.html_url,
                "statuses": [
                    {
                        "context": s.context,
                        "state": s.state,
                        "description": s.description
                    }
                    for s in statuses
                ],
                "check_runs": [
                    {
                        "name": c.name,
                        "status": c.status,
                        "conclusion": c.conclusion
                    }
                    for c in check_runs
                ],
                "reviews": [
                    {
                        "user": r.user.login,
                        "state": r.state
                    }
                    for r in pr.get_reviews()
                ]
            }
            
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            raise HTTPException(status_code=e.status, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to get PR status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "git_integration",
                "status": "healthy",
                "github_configured": self.github_client is not None,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/branch/create")
        async def create_branch(request: BranchRequest):
            """Create a new branch"""
            return await self.create_branch(
                request.repo_path,
                request.branch_name,
                request.from_branch or "main"
            )
        
        @self.app.post("/commit")
        async def commit(request: CommitRequest):
            """Commit changes"""
            return await self.commit_changes(
                request.repo_path,
                request.message,
                request.files,
                request.branch
            )
        
        @self.app.post("/push")
        async def push(repo_path: str, branch: str, remote: str = "origin"):
            """Push branch to remote"""
            return await self.push_branch(repo_path, branch, remote)
        
        @self.app.post("/pr/create")
        async def create_pr(request: PRRequest):
            """Create pull request"""
            return await self.create_pull_request(request)
        
        @self.app.post("/pr/comment")
        async def add_comment(comment: ReviewComment):
            """Add review comment to PR"""
            return await self.add_review_comment(comment)
        
        @self.app.post("/pr/{pr_number}/merge")
        async def merge_pr(pr_number: int, repo_path: str, 
                          commit_message: Optional[str] = None,
                          merge_method: str = "merge"):
            """Merge pull request"""
            return await self.merge_pull_request(
                pr_number, repo_path, commit_message, merge_method
            )
        
        @self.app.get("/pr/{pr_number}/status")
        async def get_pr_status(pr_number: int, repo_path: str):
            """Get PR status"""
            return await self.get_pr_status(pr_number, repo_path)
        
        @self.app.get("/repo/{repo_path:path}/status")
        async def get_repo_status(repo_path: str):
            """Get repository status"""
            try:
                repo = self._get_repo(repo_path)
                
                # Get current branch
                current_branch = repo.active_branch.name
                
                # Get uncommitted changes
                changed_files = [item.a_path for item in repo.index.diff(None)]
                untracked_files = repo.untracked_files
                
                # Get recent commits
                commits = []
                for commit in repo.iter_commits(max_count=10):
                    commits.append({
                        "sha": commit.hexsha[:8],
                        "message": commit.message.strip(),
                        "author": str(commit.author),
                        "date": datetime.fromtimestamp(commit.committed_date).isoformat()
                    })
                
                return {
                    "current_branch": current_branch,
                    "changed_files": changed_files,
                    "untracked_files": untracked_files,
                    "recent_commits": commits,
                    "remotes": [r.name for r in repo.remotes]
                }
                
            except Exception as e:
                logger.error(f"Failed to get repo status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/repo/clone")
        async def clone_repo(url: str, path: str, branch: Optional[str] = None):
            """Clone a repository"""
            try:
                # Clone repository
                repo = git.Repo.clone_from(url, path, branch=branch)
                
                logger.info(f"Cloned repository from {url} to {path}")
                
                return {
                    "status": "success",
                    "path": path,
                    "url": url,
                    "branch": repo.active_branch.name
                }
                
            except Exception as e:
                logger.error(f"Failed to clone repo: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Git Integration Service starting up...")
        
        # Test GitHub connection if token is available
        if self.github_client:
            try:
                user = self.github_client.get_user()
                logger.info(f"GitHub authenticated as: {user.login}")
            except Exception as e:
                logger.warning(f"GitHub authentication failed: {e}")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Git Integration Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = GitIntegrationService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("GIT_INTEGRATION_PORT", 8019))
    logger.info(f"Starting Git Integration Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()