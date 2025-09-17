__version__ = "0.1.2"

from .client import AgentLightningClient, DevTaskLoader
from .config import lightning_cli
from .litagent import LitAgent
from .llm_providers import (
    LLMProviderManager,
    OpenAIProvider,
    AnthropicProvider,
    GrokProvider,
    generate_with_model,
    get_available_providers,
    list_supported_models,
    llm_manager
)
from .logging import configure_logger
from .oauth import (
    OAuth2Manager,
    OAuth2Config,
    get_current_user,
    require_scope,
    require_role,
    require_read,
    require_write,
    require_admin,
    oauth_manager
)
from .rbac import (
    RBACManager,
    Role,
    Permission,
    rbac_manager,
    require_server_read,
    require_server_write,
    require_server_admin,
    require_task_read,
    require_task_write,
    require_task_admin,
    require_resource_read,
    require_resource_write,
    require_rollout_read,
    require_rollout_write,
    require_admin_role,
    require_manager_or_higher,
    get_user_permissions,
    get_user_role
)
from .reward import reward
from .server import AgentLightningServer
from .trainer import Trainer
from .types import *

# RL Orchestrator integration
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / 'rl_orch'))
    from rl_orch.core.orchestrator import RLOrchestrator
    from rl_orch.core.config_models import ExperimentConfig
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    RLOrchestrator = None
    ExperimentConfig = None
