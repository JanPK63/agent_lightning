# Database Encryption Strategy Design

## Overview
This document outlines the comprehensive encryption strategy for protecting sensitive data at rest in the Agent Lightning database. The strategy implements field-level encryption using AES-256 with a robust key management system.

## 🎯 Objectives
- **Data Protection**: Encrypt sensitive data at rest to prevent unauthorized access
- **Compliance**: Meet security standards and regulatory requirements
- **Performance**: Minimize performance impact on database operations
- **Scalability**: Support key rotation and management at scale
- **Auditability**: Track encryption operations and key usage

## 🔐 Encryption Algorithm Selection

### Primary Algorithm: AES-256-GCM
- **Algorithm**: Advanced Encryption Standard (AES) with 256-bit keys
- **Mode**: Galois/Counter Mode (GCM) for authenticated encryption
- **Rationale**:
  - Industry standard for data encryption
  - Provides both confidentiality and integrity
  - Resistant to known attacks
  - Hardware acceleration support on modern CPUs
  - FIPS 140-2 compliant

### Alternative: ChaCha20-Poly1305
- **Fallback Algorithm**: For systems without AES hardware acceleration
- **Mode**: Authenticated encryption with Poly1305 MAC
- **Use Case**: Mobile/edge deployments, legacy systems

## 📊 Field-Level vs Table-Level Encryption

### Decision: Field-Level Encryption
**Selected Approach**: Field-level encryption with selective application

#### Rationale for Field-Level Encryption:
- **Granular Control**: Encrypt only sensitive fields, not entire tables
- **Performance**: Better query performance on non-sensitive data
- **Flexibility**: Different encryption keys for different data types
- **Compliance**: Meet specific regulatory requirements per field
- **Maintenance**: Easier to add/remove encryption on specific fields

#### Fields Requiring Encryption:
1. **High Sensitivity**:
   - `api_keys.key_hash` (already hashed, but add additional encryption layer)
   - `users.password_hash` (add encryption wrapper)
   - `users.email` (PII - Personally Identifiable Information)
   - `conversations.user_query` (may contain sensitive user data)
   - `conversations.agent_response` (may contain sensitive AI outputs)

2. **Medium Sensitivity**:
   - `agents.config` (may contain API keys, credentials)
   - `workflows.context` (may contain sensitive workflow data)
   - `knowledge.content` (may contain sensitive learned information)

3. **Low Sensitivity**:
   - `metrics.tags` (may contain identifying information)
   - `sessions.data` (session data may be sensitive)

## 🗝️ Key Management System Design

### Key Hierarchy
```
Master Key (Environment-Specific)
├── Data Encryption Keys (DEK) - Per Table
│   ├── Field Encryption Keys (FEK) - Per Sensitive Field
│   │   ├── Record Encryption Keys (REK) - Per Database Record (Optional)
```

### Key Types

#### 1. Master Key (MK)
- **Purpose**: Encrypts all Data Encryption Keys
- **Storage**: External secure storage (AWS KMS, HashiCorp Vault, Azure Key Vault)
- **Rotation**: Annual rotation with 30-day transition period
- **Backup**: Encrypted backup with separate key

#### 2. Data Encryption Keys (DEK)
- **Purpose**: Encrypts data in specific tables
- **Scope**: One DEK per table containing sensitive data
- **Storage**: Encrypted with Master Key in database
- **Rotation**: Quarterly rotation
- **Derivation**: HKDF-SHA256 from Master Key + Table Salt

#### 3. Field Encryption Keys (FEK)
- **Purpose**: Encrypts specific sensitive fields
- **Scope**: One FEK per sensitive field
- **Storage**: Encrypted with DEK in database
- **Rotation**: Monthly rotation
- **Derivation**: HKDF-SHA256 from DEK + Field Salt

#### 4. Record Encryption Keys (REK) - Optional
- **Purpose**: Unique encryption per database record
- **Scope**: One REK per database record (for maximum security)
- **Storage**: Derived on-the-fly from FEK + Record ID
- **Rotation**: Inherited from FEK rotation

### Key Rotation Strategy

#### Automatic Rotation Schedule:
- **Master Key**: Annual (365 days)
- **Data Encryption Keys**: Quarterly (90 days)
- **Field Encryption Keys**: Monthly (30 days)

#### Rotation Process:
1. **Generate New Key**: Create new key with HKDF
2. **Dual Encryption**: Encrypt data with both old and new keys
3. **Migration Window**: 30-day transition period
4. **Cleanup**: Remove old key after successful migration
5. **Audit**: Log all rotation events

#### Emergency Key Rotation:
- **Trigger**: Security breach, key compromise
- **Process**: Immediate rotation with priority queuing
- **Notification**: Alert all system administrators

## 🔒 Secure Key Storage

### Primary Storage: External KMS
- **AWS KMS**: For AWS deployments
- **Azure Key Vault**: For Azure deployments
- **HashiCorp Vault**: For on-premises/multi-cloud
- **Google Cloud KMS**: For GCP deployments

### Fallback Storage: Encrypted Database
- **Encryption**: Master Key encrypted with environment-specific passphrase
- **Access Control**: Restricted to encryption service only
- **Monitoring**: All access attempts logged and monitored

### Key Storage Schema:
```sql
CREATE TABLE encryption_keys (
    id UUID PRIMARY KEY,
    key_type VARCHAR(20) NOT NULL, -- 'master', 'dek', 'fek'
    key_id VARCHAR(100) NOT NULL, -- Unique identifier
    encrypted_key BYTEA NOT NULL, -- Encrypted key data
    key_hash VARCHAR(64) NOT NULL, -- SHA256 hash for integrity
    algorithm VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    rotation_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);
```

## ⚡ Performance Considerations

### Encryption Overhead:
- **AES-256-GCM**: ~10-15% performance overhead
- **Key Derivation**: Minimal impact (HKDF-SHA256)
- **Database Operations**: Additional CPU for encrypt/decrypt

### Optimization Strategies:
1. **Caching**: Cache decrypted keys in memory (with TTL)
2. **Batch Operations**: Encrypt/decrypt multiple fields together
3. **Hardware Acceleration**: Use AES-NI instructions when available
4. **Connection Pooling**: Reuse encrypted connections

### Performance Benchmarks (Estimated):
- **Read Operations**: 5-10% slower
- **Write Operations**: 15-20% slower
- **Key Derivation**: <1ms per operation
- **Memory Usage**: +50MB for key cache

## 🏗️ Implementation Architecture

### Core Components:

#### 1. Encryption Service (`services/encryption_service.py`)
```python
class EncryptionService:
    def encrypt_field(self, plaintext: str, field_key: str) -> str:
        """Encrypt a field value"""

    def decrypt_field(self, ciphertext: str, field_key: str) -> str:
        """Decrypt a field value"""

    def rotate_key(self, key_type: str, key_id: str) -> bool:
        """Rotate an encryption key"""

    def derive_field_key(self, table_key: str, field_name: str) -> str:
        """Derive field-specific key from table key"""
```

#### 2. Key Management Service (`services/key_management_service.py`)
```python
class KeyManagementService:
    def get_master_key(self) -> str:
        """Retrieve master key from secure storage"""

    def generate_data_key(self, table_name: str) -> str:
        """Generate table-specific data key"""

    def rotate_keys(self, key_type: str) -> bool:
        """Rotate keys of specified type"""

    def audit_key_access(self, key_id: str, operation: str):
        """Log key access for audit trail"""
```

#### 3. SQLAlchemy Integration (`shared/encrypted_fields.py`)
```python
class EncryptedString(TypeDecorator):
    """SQLAlchemy type for encrypted string fields"""

    def process_bind_param(self, value, dialect):
        """Encrypt value before storing"""

    def process_result_value(self, value, dialect):
        """Decrypt value when retrieving"""
```

### Database Middleware:
```python
class EncryptionMiddleware:
    """Database middleware for transparent encryption/decryption"""

    def before_insert(self, mapper, connection, target):
        """Encrypt sensitive fields before insert"""

    def after_load(self, target):
        """Decrypt sensitive fields after load"""
```

## 🔍 Monitoring and Audit

### Encryption Metrics:
- **Encryption Operations**: Count of encrypt/decrypt operations
- **Key Rotations**: Success/failure rates
- **Performance**: Encryption operation latency
- **Errors**: Encryption failures and recovery

### Audit Logging:
- **Key Access**: All key retrieval operations
- **Encryption Events**: Field encryption/decryption
- **Key Rotations**: Rotation events with timestamps
- **Security Events**: Failed decryption attempts, key compromise alerts

### Monitoring Dashboard:
- Real-time encryption operation metrics
- Key rotation status and schedules
- Security alerts and incidents
- Performance impact monitoring

## 🚀 Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Implement encryption service
- Set up key management system
- Create encrypted field types
- Add basic monitoring

### Phase 2: Field Encryption (Week 3-4)
- Encrypt high-sensitivity fields
- Update SQLAlchemy models
- Implement middleware
- Add comprehensive tests

### Phase 3: Key Rotation (Week 5-6)
- Implement automatic key rotation
- Add rotation scheduling
- Create migration scripts
- Test rotation procedures

### Phase 4: Production Rollout (Week 7-8)
- Gradual encryption rollout
- Performance optimization
- Documentation updates
- Security review

## 🧪 Testing Strategy

### Unit Tests:
- Encryption/decryption functions
- Key derivation algorithms
- SQLAlchemy integration
- Middleware functionality

### Integration Tests:
- Full database operations with encryption
- Key rotation procedures
- Performance benchmarks
- Failure recovery scenarios

### Security Tests:
- Encryption strength validation
- Key compromise simulation
- Access control verification
- Audit log integrity

## 📚 Security Best Practices

### Key Management:
- **Principle of Least Privilege**: Minimal key access
- **Key Separation**: Different keys for different data types
- **Regular Rotation**: Automated key rotation schedules
- **Secure Storage**: External KMS with access controls

### Operational Security:
- **Audit Everything**: Comprehensive logging of all operations
- **Fail-Safe Design**: Graceful degradation on encryption failures
- **Backup Security**: Encrypted backups with separate keys
- **Incident Response**: Clear procedures for key compromise

### Compliance:
- **Data Classification**: Clear classification of sensitive data
- **Regulatory Compliance**: Meet GDPR, HIPAA, SOC 2 requirements
- **Access Controls**: Role-based access to encrypted data
- **Retention Policies**: Secure deletion of old encryption keys

## 📋 Risk Assessment

### High-Risk Scenarios:
1. **Master Key Compromise**: Complete system compromise
2. **Encryption Service Failure**: Data becomes inaccessible
3. **Key Rotation Failure**: Stale keys reduce security
4. **Performance Degradation**: System slowdown affects availability

### Mitigation Strategies:
1. **Master Key**: Multi-party key custody, regular rotation
2. **Service Failure**: Redundant encryption services, fallback modes
3. **Rotation Failure**: Automated retry mechanisms, manual override
4. **Performance**: Hardware acceleration, caching, optimization

## 🎯 Success Criteria

### Security Objectives:
- ✅ All sensitive data encrypted at rest
- ✅ Encryption keys properly managed and rotated
- ✅ Comprehensive audit trail maintained
- ✅ Compliance with security standards met

### Performance Objectives:
- ✅ <15% performance degradation on database operations
- ✅ <5% increase in memory usage
- ✅ <1 second key rotation time
- ✅ 99.9% encryption operation success rate

### Operational Objectives:
- ✅ Automated key rotation processes
- ✅ Real-time monitoring and alerting
- ✅ Comprehensive documentation
- ✅ Zero data loss during encryption rollout

---

**Document Version**: 1.0
**Last Updated**: 2025-09-17
**Review Date**: Monthly
**Owner**: Security Team