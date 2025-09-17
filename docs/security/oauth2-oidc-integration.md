# OAuth2/OIDC Integration for Enterprise SSO

Agent Lightning now supports enterprise-grade authentication through OAuth2/OpenID Connect (OIDC) integration, enabling seamless single sign-on (SSO) with popular identity providers.

## Overview

The OAuth2/OIDC integration provides:
- **Enterprise SSO** with major identity providers
- **Multi-provider support** (Google, Microsoft, Okta, Auth0, Keycloak)
- **Comprehensive audit logging** for security compliance
- **Role-based access control (RBAC)** integration
- **JWT token management** with automatic refresh
- **Session management** with configurable timeouts

## Supported Identity Providers

### Google OAuth2
```bash
# Environment Variables
GOOGLE_OAUTH_CLIENT_ID=your_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret
GOOGLE_OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
```

**Setup Steps:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select a project
3. Enable Google+ API
4. Create OAuth 2.0 Client ID credentials
5. Add authorized redirect URI: `http://localhost:8000/auth/callback`

### Microsoft Azure AD
```bash
# Environment Variables
MICROSOFT_OAUTH_CLIENT_ID=your_client_id
MICROSOFT_OAUTH_CLIENT_SECRET=your_client_secret
MICROSOFT_OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
```

**Setup Steps:**
1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to Azure Active Directory > App registrations
3. Create new registration
4. Add redirect URI: `http://localhost:8000/auth/callback`
5. Note Application (client) ID and create client secret

### Okta
```bash
# Environment Variables
OKTA_BASE_URL=https://your-org.okta.com
OKTA_OAUTH_CLIENT_ID=your_client_id
OKTA_OAUTH_CLIENT_SECRET=your_client_secret
OKTA_OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
```

**Setup Steps:**
1. Go to Okta Admin Console
2. Navigate to Applications > Create App Integration
3. Choose OIDC and Web Application
4. Add sign-in redirect URI: `http://localhost:8000/auth/callback`

### Auth0
```bash
# Environment Variables
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_REDIRECT_URI=http://localhost:8000/auth/callback
```

**Setup Steps:**
1. Go to Auth0 Dashboard
2. Create a new Regular Web Application
3. Configure allowed callback URLs
4. Note Domain, Client ID, and Client Secret

### Keycloak
```bash
# Environment Variables
KEYCLOAK_BASE_URL=http://localhost:8080
KEYCLOAK_REALM=your-realm
KEYCLOAK_CLIENT_ID=your_client_id
KEYCLOAK_CLIENT_SECRET=your_client_secret
KEYCLOAK_REDIRECT_URI=http://localhost:8000/auth/callback
```

**Setup Steps:**
1. Access Keycloak Admin Console
2. Create/select a realm
3. Go to Clients > Create client
4. Configure client settings
5. Add redirect URI: `http://localhost:8000/auth/callback`

## API Endpoints

### Initiate OAuth2 Login
```http
GET /auth/oauth/login?state=optional_state
```

**Response:**
```json
{
  "authorization_url": "https://auth.provider.com/oauth/authorize?client_id=...&response_type=code&scope=openid profile email&redirect_uri=...&state=..."
}
```

### OAuth2 Callback
```http
GET /auth/oauth/callback?code=auth_code&state=state_value
```

**Response:**
```json
{
  "access_token": "jwt_token_here",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user_info": {
    "sub": "user_id",
    "email": "user@example.com",
    "name": "User Name",
    "groups": ["group1", "group2"],
    "roles": ["developer"]
  },
  "session_id": "session_uuid"
}
```

### List Available Providers
```http
GET /auth/oauth/providers
```

**Response:**
```json
{
  "oauth_enabled": true,
  "authorization_url": "https://auth.provider.com/oauth/authorize?...",
  "available_providers": {
    "google": true,
    "microsoft": false,
    "okta": true
  },
  "supported_providers": ["google", "microsoft", "okta", "auth0", "keycloak"]
}
```

### Provider Setup Instructions
```http
GET /auth/oauth/providers/{provider_name}/setup
```

**Response:**
```json
{
  "provider": "google",
  "setup_instructions": "Go to Google Cloud Console...\n1. Create/select a project\n2. Enable Google+ API\n...",
  "required_env_vars": {
    "client_id_env": "GOOGLE_OAUTH_CLIENT_ID",
    "client_secret_env": "GOOGLE_OAUTH_CLIENT_SECRET",
    "redirect_uri_env": "GOOGLE_OAUTH_REDIRECT_URI"
  }
}
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OAUTH2_PROVIDER_URL` | Base URL of OAuth2 provider | Yes* |
| `OAUTH2_CLIENT_ID` | OAuth2 client ID | Yes |
| `OAUTH2_CLIENT_SECRET` | OAuth2 client secret | Yes |
| `OAUTH2_REDIRECT_URI` | OAuth2 redirect URI | No (defaults to localhost:8000) |
| `OAUTH2_TOKEN_URL` | OAuth2 token endpoint | No (auto-detected) |
| `OAUTH2_AUTHORIZATION_URL` | OAuth2 authorization endpoint | No (auto-detected) |
| `OAUTH2_USERINFO_URL` | OAuth2 userinfo endpoint | No (auto-detected) |
| `OIDC_ISSUER` | OIDC issuer URL | No (auto-detected) |
| `OIDC_JWKS_URI` | OIDC JWKS endpoint | No (auto-detected) |
| `OAUTH2_SCOPES` | OAuth2 scopes (space-separated) | No (defaults to "openid profile email") |
| `JWT_SECRET` | JWT signing secret | No (auto-generated) |
| `JWT_ALGORITHM` | JWT algorithm | No (defaults to HS256) |
| `JWT_EXPIRATION_HOURS` | JWT expiration time | No (defaults to 24) |

*Required when using generic OAuth2 (not pre-configured providers)

### Provider-Specific Variables

Each supported provider has its own set of environment variables as shown in the provider sections above.

## Security Features

### Audit Logging
All authentication events are logged with comprehensive details:
- User actions (login, logout, registration)
- Authentication attempts (success/failure)
- Security events (failed attempts, suspicious activity)
- IP addresses and user agents
- Timestamps and session information

### Rate Limiting
Built-in rate limiting protects against:
- Brute force attacks
- Excessive API calls
- Suspicious login patterns

### Session Management
- Configurable session timeouts
- Automatic session cleanup
- Secure session storage in Redis
- Session invalidation on logout

### Token Security
- JWT tokens with configurable expiration
- Automatic token refresh
- Secure token storage
- Token blacklisting on logout

## Integration Examples

### Python Client
```python
import requests

# Initiate OAuth2 login
response = requests.get("http://localhost:8000/auth/oauth/login")
auth_url = response.json()["authorization_url"]

# Redirect user to auth_url
# User authenticates with provider
# Provider redirects to /auth/oauth/callback

# Handle callback
callback_response = requests.get("http://localhost:8000/auth/oauth/callback", params={
    "code": "auth_code_from_provider",
    "state": "state_value"
})

tokens = callback_response.json()
access_token = tokens["access_token"]

# Use access token for authenticated requests
headers = {"Authorization": f"Bearer {access_token}"}
api_response = requests.get("http://localhost:8000/api/protected", headers=headers)
```

### JavaScript Client
```javascript
// Initiate OAuth2 login
const response = await fetch('/auth/oauth/login');
const { authorization_url } = await response.json();

// Redirect to authorization URL
window.location.href = authorization_url;

// Handle callback (in callback page)
const urlParams = new URLSearchParams(window.location.search);
const code = urlParams.get('code');
const state = urlParams.get('state');

const tokenResponse = await fetch('/auth/oauth/callback', {
  method: 'GET',
  headers: {
    'Content-Type': 'application/json',
  },
  // Note: In production, handle code/state server-side
});

// Store tokens securely
const tokens = await tokenResponse.json();
localStorage.setItem('access_token', tokens.access_token);
```

## Troubleshooting

### Common Issues

1. **"OAuth2 not configured" error**
   - Ensure required environment variables are set
   - Check provider configuration
   - Verify client ID and secret are correct

2. **"Invalid token" error**
   - Check JWT secret consistency
   - Verify token hasn't expired
   - Ensure proper token format

3. **Callback URL mismatch**
   - Ensure redirect URI matches provider configuration
   - Check for URL encoding issues
   - Verify callback endpoint is accessible

### Debug Mode
Enable debug logging to troubleshoot issues:
```bash
export OAUTH2_DEBUG=true
```

### Health Checks
Monitor OAuth2 integration health:
```http
GET /health
```

Response includes OAuth2 status:
```json
{
  "service": "auth",
  "oauth_enabled": true,
  "providers_configured": ["google", "microsoft"],
  "audit_logging": "active"
}
```

## Best Practices

1. **Use HTTPS in production** - Always configure HTTPS for OAuth2 callbacks
2. **Store secrets securely** - Use environment variables or secure secret management
3. **Implement proper error handling** - Handle OAuth2 errors gracefully
4. **Monitor audit logs** - Regularly review authentication logs for security
5. **Configure appropriate scopes** - Request only necessary OAuth2 scopes
6. **Set reasonable token expiration** - Balance security with user experience
7. **Implement logout properly** - Clear sessions and tokens on logout
8. **Use state parameter** - Always include state parameter for CSRF protection

## Migration from Password Authentication

Existing password-based authentication continues to work alongside OAuth2/OIDC. Users can choose their preferred authentication method.

To migrate existing users:
1. Users can link OAuth2 accounts to existing profiles
2. Gradual migration with fallback to password authentication
3. Clear communication about OAuth2 benefits

## Compliance

The OAuth2/OIDC integration supports:
- **SOC 2** compliance with comprehensive audit logging
- **GDPR** compliance with proper data handling
- **Enterprise security** requirements
- **Multi-tenant** architectures
- **Regulatory compliance** reporting

## Support

For issues with OAuth2/OIDC integration:
1. Check provider-specific documentation
2. Review audit logs for error details
3. Verify environment variable configuration
4. Test with provider's OAuth2 playground
5. Contact support with debug logs

---

*Last updated: 2025-09-17*