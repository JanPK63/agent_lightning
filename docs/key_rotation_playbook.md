# Key Rotation Playbook

## Overview
This playbook outlines the safe execution of encryption key rotation for Agent Lightning's database encryption system. Key rotation reduces blast radius by limiting exposure windows and meets compliance requirements.

## Key Rotation Strategy
- **Data Keys**: Rotate every 90 days
- **Field Keys**: Rotate every 30 days
- **Algorithm**: AES-256-GCM with authenticated encryption
- **Hierarchy**: Master → Data → Field keys

## Execution Phases

### Phase 1: Staging Full Rotation
**Purpose**: Validate rotation process in staging environment
**Scope**: All data and field keys in staging database
**Duration**: ~2-4 hours depending on dataset size
**Success Criteria**:
- Decrypt error rate < 0.1%
- No audit log gaps
- All encrypted fields accessible
- Performance impact < 15%

### Phase 2: Production Canary
**Purpose**: Test rotation in production with minimal risk
**Scope**: 1% of production records (random sample)
**Duration**: ~1-2 hours
**Success Criteria**:
- Decrypt error rate < 0.1%
- No user-facing errors
- Audit logs complete
- 24h observation period

### Phase 3: Production Staged Rollout
**Purpose**: Gradual production rollout with rollback capability
**Scope**: 25% → 50% → 75% → 100% of records
**Duration**: ~4-8 hours total
**Success Criteria**:
- Each stage: Decrypt error rate < 0.1%
- 2h observation between stages
- Rollback capability maintained

## Pre-Rotation Checklist

### Environment Preparation
- [ ] Database backups completed (last 24h)
- [ ] Monitoring dashboards enabled
- [ ] Alert thresholds configured
- [ ] On-call engineer available
- [ ] Rollback plan documented

### Key Management
- [ ] New keys generated and stored
- [ ] Old keys accessible for rollback
- [ ] Key rotation policy active
- [ ] Audit logging enabled

### Testing
- [ ] Rotation scripts tested in staging
- [ ] Monitoring alerts tested
- [ ] Rollback procedure tested
- [ ] Communication plan ready

## Rotation Execution Steps

### Step 1: Pre-Rotation Validation
```bash
# Verify system health
./scripts/health_check.sh

# Check key status
python -c "from services.key_management_service import KeyManagementService; kms = KeyManagementService(); print(kms.get_key_status())"

# Validate audit logging
python -c "from shared.database import get_session; session = get_session(); count = session.query(KeyUsageLog).count(); print(f'Audit logs: {count}')"
```

### Step 2: Generate New Keys
```python
from services.key_management_service import KeyManagementService

kms = KeyManagementService()
new_data_key = kms.generate_key('data', 'rotation_data_key')
new_field_key = kms.generate_key('field', 'rotation_field_key')

print(f"New data key: {new_data_key.id}")
print(f"New field key: {new_field_key.id}")
```

### Step 3: Execute Rotation
```python
from services.key_rotation_service import KeyRotationService

rotation_service = KeyRotationService()
result = rotation_service.rotate_keys(
    data_key_id=new_data_key.id,
    field_key_id=new_field_key.id,
    batch_size=1000,
    dry_run=False
)

print(f"Rotation completed: {result}")
```

### Step 4: Post-Rotation Validation
```python
# Check decrypt error rate
from shared.database import get_session
session = get_session()

errors = session.query(KeyUsageLog).filter(
    KeyUsageLog.operation == 'decrypt',
    KeyUsageLog.success == False
).count()

total_decrypts = session.query(KeyUsageLog).filter(
    KeyUsageLog.operation == 'decrypt'
).count()

error_rate = errors / total_decrypts if total_decrypts > 0 else 0
print(f"Decrypt error rate: {error_rate:.4f}")

# Validate data integrity
from shared.models import User, Agent, Conversation
user_count = session.query(User).count()
agent_count = session.query(Agent).count()
conv_count = session.query(Conversation).count()

print(f"Users: {user_count}, Agents: {agent_count}, Conversations: {conv_count}")
```

## Monitoring and Alerts

### SLOs (Service Level Objectives)
- **Decrypt Error Rate**: < 0.1% of all decrypt operations
- **Rotation Duration**: < 4 hours for full rotation
- **Audit Coverage**: 100% of encryption operations logged
- **Performance Impact**: < 15% increase in query latency

### Alert Thresholds
- Decrypt error rate > 0.1%: Page immediately
- Audit log gaps > 5 minutes: Page immediately
- Rotation job failure: Page immediately
- Performance degradation > 20%: Warn

### Monitoring Dashboards
- Grafana: Encryption Health Dashboard
- Metrics: decrypt_errors_total, key_rotation_duration, audit_log_gaps
- Logs: Key rotation events, error details

## Rollback Procedure

### Immediate Rollback (< 30 minutes)
If SLOs breached within first 30 minutes:
1. Stop rotation job
2. Switch back to previous active keys
3. Re-encrypt affected records with old keys
4. Validate system health
5. Notify stakeholders

### Delayed Rollback (30min - 24h)
If issues discovered during observation:
1. Assess impact and urgency
2. Schedule maintenance window
3. Execute rollback steps
4. Full system validation
5. Incident review

### Emergency Rollback
If critical system impact:
1. Immediate page to on-call
2. Database restore from backup if needed
3. Full incident response protocol

## Communication Plan

### Pre-Rotation
- [ ] Engineering team notified 24h in advance
- [ ] Stakeholders informed of maintenance window
- [ ] Status page updated if applicable

### During Rotation
- [ ] Real-time status updates to engineering team
- [ ] Alerts for any SLO breaches
- [ ] Stakeholder updates for production rotations

### Post-Rotation
- [ ] Success confirmation to all stakeholders
- [ ] Incident review if issues occurred
- [ ] Documentation updates

## Evidence Collection

### Rotation Artifacts
- Rotation start/end timestamps
- Key IDs before/after
- Error logs and metrics
- Performance impact measurements
- Audit log completeness verification

### Storage
- Artifacts stored in `exports/rotation_reports/`
- JSON format with full metadata
- Retention: 1 year minimum

## Risk Mitigation

### Technical Risks
- **Data Loss**: Pre-rotation backups required
- **Performance Impact**: Load testing in staging
- **Key Compromise**: Secure key generation and storage
- **Audit Gaps**: Redundant logging systems

### Operational Risks
- **Incomplete Rotation**: Batch processing with checkpoints
- **Monitoring Blind Spots**: Multiple monitoring systems
- **Communication Failures**: Multiple notification channels
- **Rollback Complexity**: Tested rollback procedures

## Success Criteria

### Technical Success
- [ ] All encrypted fields accessible
- [ ] No data corruption detected
- [ ] Performance within acceptable bounds
- [ ] Audit logs complete and accurate

### Operational Success
- [ ] No production incidents
- [ ] Rollback capability maintained
- [ ] Stakeholders informed appropriately
- [ ] Documentation updated

### Compliance Success
- [ ] Rotation logged for audit purposes
- [ ] Key lifecycle properly managed
- [ ] Security requirements met

## Post-Rotation Tasks

### Immediate (Next Day)
- [ ] Validate system stability
- [ ] Update rotation schedules
- [ ] Archive rotation artifacts
- [ ] Update documentation

### Short Term (Next Week)
- [ ] Review monitoring effectiveness
- [ ] Update alert thresholds if needed
- [ ] Train team on procedures
- [ ] Schedule next rotation

### Long Term (Next Month)
- [ ] Review rotation frequency
- [ ] Assess automation opportunities
- [ ] Update compliance documentation
- [ ] Plan next major version

## Contacts

### Primary
- **Lead Engineer**: [Name] - [Contact]
- **Security Officer**: [Name] - [Contact]
- **On-call Engineer**: Current rotation

### Secondary
- **DevOps Team**: [Slack Channel]
- **Security Team**: [Email/Slack]
- **Management**: [Email]

## References

- [Encryption Strategy Design](encryption_strategy_design.md)
- [Key Management Service Documentation](../services/key_management_service.py)
- [Database Migration Scripts](../migrations/)
- [Monitoring Dashboard Configuration](../grafana/dashboards/)