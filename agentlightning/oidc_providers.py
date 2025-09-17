#!/usr/bin/env python3
"""
OIDC Provider Configurations for Agent Lightning
Pre-configured settings for popular OIDC providers
"""

import os
from typing import Dict, Any


class OIDCProviderConfig:
    """Configuration for OIDC providers"""

    PROVIDERS = {
        "google": {
            "name": "Google",
            "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
            "issuer": "https://accounts.google.com",
            "scopes": ["openid", "profile", "email"],
            "client_id_env": "GOOGLE_OAUTH_CLIENT_ID",
            "client_secret_env": "GOOGLE_OAUTH_CLIENT_SECRET",
            "redirect_uri_env": "GOOGLE_OAUTH_REDIRECT_URI"
        },
        "microsoft": {
            "name": "Microsoft",
            "authorization_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "userinfo_url": "https://graph.microsoft.com/oidc/userinfo",
            "jwks_uri": "https://login.microsoftonline.com/common/discovery/v2.0/keys",
            "issuer": "https://login.microsoftonline.com/common/v2.0",
            "scopes": ["openid", "profile", "email"],
            "client_id_env": "MICROSOFT_OAUTH_CLIENT_ID",
            "client_secret_env": "MICROSOFT_OAUTH_CLIENT_SECRET",
            "redirect_uri_env": "MICROSOFT_OAUTH_REDIRECT_URI"
        },
        "okta": {
            "name": "Okta",
            "authorization_url": "{base_url}/oauth2/v1/authorize",
            "token_url": "{base_url}/oauth2/v1/token",
            "userinfo_url": "{base_url}/oauth2/v1/userinfo",
            "jwks_uri": "{base_url}/oauth2/v1/keys",
            "issuer": "{base_url}/oauth2/default",
            "scopes": ["openid", "profile", "email"],
            "client_id_env": "OKTA_OAUTH_CLIENT_ID",
            "client_secret_env": "OKTA_OAUTH_CLIENT_SECRET",
            "redirect_uri_env": "OKTA_OAUTH_REDIRECT_URI",
            "base_url_env": "OKTA_BASE_URL"
        },
        "auth0": {
            "name": "Auth0",
            "authorization_url": "https://{domain}/authorize",
            "token_url": "https://{domain}/oauth/token",
            "userinfo_url": "https://{domain}/userinfo",
            "jwks_uri": "https://{domain}/.well-known/jwks.json",
            "issuer": "https://{domain}/",
            "scopes": ["openid", "profile", "email"],
            "client_id_env": "AUTH0_CLIENT_ID",
            "client_secret_env": "AUTH0_CLIENT_SECRET",
            "redirect_uri_env": "AUTH0_REDIRECT_URI",
            "domain_env": "AUTH0_DOMAIN"
        },
        "keycloak": {
            "name": "Keycloak",
            "authorization_url": "{base_url}/realms/{realm}/protocol/openid-connect/auth",
            "token_url": "{base_url}/realms/{realm}/protocol/openid-connect/token",
            "userinfo_url": "{base_url}/realms/{realm}/protocol/openid-connect/userinfo",
            "jwks_uri": "{base_url}/realms/{realm}/protocol/openid-connect/certs",
            "issuer": "{base_url}/realms/{realm}",
            "scopes": ["openid", "profile", "email"],
            "client_id_env": "KEYCLOAK_CLIENT_ID",
            "client_secret_env": "KEYCLOAK_CLIENT_SECRET",
            "redirect_uri_env": "KEYCLOAK_REDIRECT_URI",
            "base_url_env": "KEYCLOAK_BASE_URL",
            "realm_env": "KEYCLOAK_REALM"
        }
    }

    @classmethod
    def get_provider_config(cls, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        if provider_name not in cls.PROVIDERS:
            raise ValueError(f"Unknown OIDC provider: {provider_name}")

        config = cls.PROVIDERS[provider_name].copy()

        # Replace template variables
        if provider_name == "okta":
            base_url = os.getenv(config["base_url_env"])
            if not base_url:
                raise ValueError(f"Missing {config['base_url_env']} environment variable")
            config["authorization_url"] = config["authorization_url"].format(base_url=base_url)
            config["token_url"] = config["token_url"].format(base_url=base_url)
            config["userinfo_url"] = config["userinfo_url"].format(base_url=base_url)
            config["jwks_uri"] = config["jwks_uri"].format(base_url=base_url)
            config["issuer"] = config["issuer"].format(base_url=base_url)

        elif provider_name == "auth0":
            domain = os.getenv(config["domain_env"])
            if not domain:
                raise ValueError(f"Missing {config['domain_env']} environment variable")
            config["authorization_url"] = config["authorization_url"].format(domain=domain)
            config["token_url"] = config["token_url"].format(domain=domain)
            config["userinfo_url"] = config["userinfo_url"].format(domain=domain)
            config["jwks_uri"] = config["jwks_uri"].format(domain=domain)
            config["issuer"] = config["issuer"].format(domain=domain)

        elif provider_name == "keycloak":
            base_url = os.getenv(config["base_url_env"])
            realm = os.getenv(config["realm_env"])
            if not base_url or not realm:
                raise ValueError(f"Missing {config['base_url_env']} or {config['realm_env']} environment variables")
            config["authorization_url"] = config["authorization_url"].format(base_url=base_url, realm=realm)
            config["token_url"] = config["token_url"].format(base_url=base_url, realm=realm)
            config["userinfo_url"] = config["userinfo_url"].format(base_url=base_url, realm=realm)
            config["jwks_uri"] = config["jwks_uri"].format(base_url=base_url, realm=realm)
            config["issuer"] = config["issuer"].format(base_url=base_url, realm=realm)

        # Get client credentials from environment
        config["client_id"] = os.getenv(config["client_id_env"])
        config["client_secret"] = os.getenv(config["client_secret_env"])
        config["redirect_uri"] = os.getenv(config["redirect_uri_env"], "http://localhost:8000/auth/callback")

        return config

    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get availability status of all providers"""
        available = {}
        for provider_name in cls.PROVIDERS.keys():
            try:
                config = cls.get_provider_config(provider_name)
                available[provider_name] = bool(
                    config.get("client_id") and config.get("client_secret")
                )
            except (ValueError, KeyError):
                available[provider_name] = False
        return available

    @classmethod
    def setup_provider(cls, provider_name: str) -> Dict[str, Any]:
        """Setup and return OAuth2 config for a provider"""
        provider_config = cls.get_provider_config(provider_name)

        # Create OAuth2 config dict
        oauth_config = {
            "provider_url": provider_config["issuer"],
            "client_id": provider_config["client_id"],
            "client_secret": provider_config["client_secret"],
            "redirect_uri": provider_config["redirect_uri"],
            "issuer": provider_config["issuer"],
            "jwks_uri": provider_config["jwks_uri"],
            "token_url": provider_config["token_url"],
            "authorization_url": provider_config["authorization_url"],
            "userinfo_url": provider_config["userinfo_url"],
            "scopes": provider_config["scopes"]
        }

        return oauth_config


def get_provider_setup_instructions(provider_name: str) -> str:
    """Get setup instructions for a provider"""
    instructions = {
        "google": """
Google OAuth2 Setup:
1. Go to Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable Google+ API
4. Go to Credentials > Create Credentials > OAuth 2.0 Client IDs
5. Set authorized redirect URI: http://localhost:8000/auth/callback
6. Set environment variables:
   - GOOGLE_OAUTH_CLIENT_ID=your_client_id
   - GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret
   - GOOGLE_OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
""",
        "microsoft": """
Microsoft OAuth2 Setup:
1. Go to Azure Portal: https://portal.azure.com/
2. Navigate to Azure Active Directory > App registrations
3. Create new registration
4. Add redirect URI: http://localhost:8000/auth/callback
5. Note the Application (client) ID and create a client secret
6. Set environment variables:
   - MICROSOFT_OAUTH_CLIENT_ID=your_client_id
   - MICROSOFT_OAUTH_CLIENT_SECRET=your_client_secret
   - MICROSOFT_OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
""",
        "okta": """
Okta OAuth2 Setup:
1. Go to Okta Admin Console
2. Navigate to Applications > Create App Integration
3. Choose OIDC and Web Application
4. Add sign-in redirect URI: http://localhost:8000/auth/callback
5. Note the Client ID and Client Secret
6. Set environment variables:
   - OKTA_BASE_URL=https://your-org.okta.com
   - OKTA_OAUTH_CLIENT_ID=your_client_id
   - OKTA_OAUTH_CLIENT_SECRET=your_client_secret
   - OKTA_OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
""",
        "auth0": """
Auth0 Setup:
1. Go to Auth0 Dashboard
2. Create a new application (Regular Web App)
3. Configure allowed callback URLs: http://localhost:8000/auth/callback
4. Note the Domain, Client ID, and Client Secret
5. Set environment variables:
   - AUTH0_DOMAIN=your-domain.auth0.com
   - AUTH0_CLIENT_ID=your_client_id
   - AUTH0_CLIENT_SECRET=your_client_secret
   - AUTH0_REDIRECT_URI=http://localhost:8000/auth/callback
""",
        "keycloak": """
Keycloak Setup:
1. Access Keycloak Admin Console
2. Create a new realm or use existing
3. Go to Clients > Create client
4. Set Client ID and configure settings
5. Add redirect URI: http://localhost:8000/auth/callback
6. Create client secret
7. Set environment variables:
   - KEYCLOAK_BASE_URL=http://localhost:8080
   - KEYCLOAK_REALM=your-realm
   - KEYCLOAK_CLIENT_ID=your_client_id
   - KEYCLOAK_CLIENT_SECRET=your_client_secret
   - KEYCLOAK_REDIRECT_URI=http://localhost:8000/auth/callback
"""
    }

    return instructions.get(provider_name, f"No setup instructions available for {provider_name}")


# Example usage
if __name__ == "__main__":
    # Print available providers
    available = OIDCProviderConfig.get_available_providers()
    print("Available OIDC Providers:")
    for provider, is_available in available.items():
        status = "✅ Configured" if is_available else "❌ Not configured"
        print(f"  {provider}: {status}")

    # Show setup instructions for a provider
    print("\nSetup instructions for Google:")
    print(get_provider_setup_instructions("google"))