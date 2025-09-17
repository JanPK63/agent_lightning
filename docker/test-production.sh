#!/bin/bash

set -e

echo "🧪 Testing Production Setup..."

# Test SSL certificates
echo "🔒 Testing SSL certificates..."
if [ -f "docker/configs/ssl/nginx.crt" ] && [ -f "docker/configs/ssl/nginx.key" ]; then
    echo "✅ SSL certificates exist"
    openssl x509 -in docker/configs/ssl/nginx.crt -text -noout | grep "Subject:" || echo "❌ Invalid certificate"
else
    echo "❌ SSL certificates missing"
fi

# Test secure Docker Compose
echo "🐳 Testing secure Docker Compose..."
docker-compose -f docker-compose.production-secure.yml config --quiet && echo "✅ Secure compose valid" || echo "❌ Secure compose invalid"

# Test deployment script
echo "🚀 Testing deployment script..."
[ -x "deploy.sh" ] && echo "✅ Deploy script executable" || echo "❌ Deploy script not executable"

# Test backup script
echo "💾 Testing backup script..."
[ -x "docker/backup/backup.sh" ] && echo "✅ Backup script executable" || echo "❌ Backup script not executable"

# Test security configurations
echo "🛡️ Testing security configurations..."
[ -f "docker/security/Dockerfile.secure" ] && echo "✅ Secure Dockerfile exists" || echo "❌ Secure Dockerfile missing"
[ -f "docker/security/secrets.yml" ] && echo "✅ Secrets config exists" || echo "❌ Secrets config missing"

echo "✅ Production setup tests complete!"