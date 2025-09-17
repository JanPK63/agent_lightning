# Encryption Runbook

## Emergency Procedures

### Decryption Failures (> 0.1% error rate)

**Immediate Actions:**
1. **Stop all encryption operations** - Disable encryption middleware
2. **Check key status** - Verify active keys are not compromised
3. **Enable emergency logging** - Increase log verbosity
4. **Page on-call security engineer**

**Investigation Steps:**
1. Check recent key rotations or deployments
2. Review audit logs for unusual patterns
3. Verify key integrity and expiration
4. Check database connectivity and performance

**Recovery Options:**
- **Option A**: Rollback to previous key version (if available)
- **Option B**: Emergency key rotation with extended observation
- **Option C**: Disable encryption temporarily (last resort)

**Rollback Procedure:**
```bash
# 1. Stop application traffic
kubectl scale deployment agent-lightning --replicas=0

# 2. Switch to previous keys
python scripts/emergency_key_rollback.py --previous-version

# 3. Re-encrypt affected data
python scripts/emergency_data_recovery.py --batch-size=100

# 4. Validate system health
python scripts/health_check.py --encryption-focus

# 5. Gradually restore traffic
kubectl scale deployment agent-lightning --replicas=1
kubectl scale deployment agent-lightning --replicas=10
```

### Key Compromise Detection

**Indicators:**
- Unusual key access patterns
- Failed integrity checks
- Audit log anomalies
- Performance degradation

**Response:**
1. **Isolate compromised keys** - Mark as compromised in database
2. **Generate emergency keys** - Create new key set immediately
3. **Emergency rotation** - Rotate all data to new keys
4. **Forensic analysis** - Preserve logs for investigation
5. **Security incident response** - Follow company IR procedures

### Audit Log Gaps

**Detection:**
- Monitoring alerts for missing audit entries
- Periodic audit log completeness checks

**Response:**
1. **Verify logging infrastructure** - Check log shipping and storage
2. **Identify gap timeframe** - Determine affected operations
3. **Manual audit reconstruction** - Reconstruct critical operations
4. **System health check** - Ensure no broader issues
5. **Preventive measures** - Implement redundant logging

## Key Rotation Procedures

### Scheduled Rotation

**Prerequisites:**
- [ ] 24h advance notice to stakeholders
- [ ] Backup validation completed
- [ ] Monitoring dashboards enabled
- [ ] On-call engineer available
- [ ] Rollback plan documented

**Execution:**
```bash
# Pre-rotation validation
python scripts/rotation_health_check.py

# Execute rotation
python scripts/execute_key_rotation.py production --data-key-id=NEW_DATA_KEY --field-key-id=NEW_FIELD_KEY

# Post-rotation monitoring (24h)
# Monitor dashboards and alerts
# Validate no SLO breaches
```

**Success Criteria:**
- [ ] Decrypt error rate < 0.1%
- [ ] No audit log gaps
- [ ] Performance impact < 15%
- [ ] All encrypted fields accessible

### Emergency Rotation

**When to Use:**
- Key compromise suspected
- Security vulnerability discovered
- Regulatory requirement
- Audit finding

**Accelerated Timeline:**
- Pre-rotation: 1 hour
- Execution: 2-4 hours
- Observation: 6 hours
- Full validation: 24 hours

**Communication:**
- Immediate notification to security team
- Stakeholder updates every 2 hours
- Incident report within 24 hours

## Monitoring and Health Checks

### Daily Health Checks

**Automated Checks:**
```bash
# Run daily encryption health check
python scripts/daily_encryption_check.py

# Verify key expiration status
python scripts/check_key_expiration.py

# Validate audit log completeness
python scripts/audit_log_integrity_check.py
```

**Manual Reviews:**
- [ ] Review encryption error trends
- [ ] Check key usage patterns
- [ ] Validate compliance with rotation policies
- [ ] Review security monitoring alerts

### Alert Response Times

| Alert Severity | Response Time | Escalation |
|---------------|---------------|------------|
| Critical | 15 minutes | Page on-call |
| Warning | 1 hour | Email notification |
| Info | 4 hours | Weekly review |

### Performance Monitoring

**Key Metrics:**
- Query latency increase (< 15%)
- CPU usage for encryption operations
- Memory usage patterns
- Database connection pool status

**Thresholds:**
- Latency increase > 20%: Warning
- CPU usage > 80%: Critical
- Memory usage > 90%: Critical
- Connection pool exhausted: Critical

## Data Recovery Procedures

### Single Record Corruption

**Detection:**
- Individual decryption failures
- Application error logs
- User reports

**Recovery:**
1. **Identify affected record** - Use error logs and audit trails
2. **Check backup integrity** - Verify record exists in backup
3. **Manual re-encryption** - Re-encrypt with current keys
4. **Validate access** - Confirm record is accessible

### Bulk Data Corruption

**Detection:**
- High error rates across multiple records
- Systematic decryption failures
- Key integrity check failures

**Recovery:**
1. **Stop all access** - Prevent further corruption
2. **Assess damage scope** - Determine affected data sets
3. **Restore from backup** - Use last known good backup
4. **Re-encryption** - Apply current encryption keys
5. **Gradual rollout** - Test with small batches first

### Complete Key Loss

**Worst Case Scenario:**
- All encryption keys lost or corrupted
- No recent backups available

**Recovery Options:**
1. **Key reconstruction** - Attempt from key derivation materials
2. **Emergency key generation** - Create new keys (data loss)
3. **Data migration** - Move to new encryption system
4. **Business continuity** - Implement compensating controls

## Compliance and Auditing

### Regulatory Requirements

**GDPR Article 32:**
- Encryption of personal data at rest
- Key management procedures
- Incident response capabilities

**SOC 2 Type II:**
- Encryption key rotation policies
- Access controls and monitoring
- Change management procedures

**PCI DSS:**
- Strong cryptography for cardholder data
- Key management procedures
- Audit trail requirements

### Audit Evidence Collection

**Required Artifacts:**
- [ ] Key rotation schedules and execution logs
- [ ] Encryption error monitoring reports
- [ ] Access control and audit logs
- [ ] Incident response documentation
- [ ] Backup and recovery test results

**Retention Period:**
- Encryption logs: 7 years
- Key rotation records: 7 years
- Incident reports: 7 years
- Audit evidence: 7 years

## Training and Documentation

### Team Training Requirements

**Required Training:**
- [ ] Encryption system overview
- [ ] Key rotation procedures
- [ ] Emergency response procedures
- [ ] Monitoring and alerting
- [ ] Compliance requirements

**Frequency:**
- Initial training: Before system access
- Refresher training: Annual
- Emergency procedures: Quarterly drills

### Documentation Updates

**When to Update:**
- After significant changes
- Following incidents
- Annual review
- Regulatory changes

**Review Process:**
1. Technical review by security team
2. Business stakeholder review
3. Legal/compliance review
4. Management approval

## Contact Information

### Primary Contacts

**Security Team:**
- Security Lead: [Name] - [Email] - [Phone]
- Encryption Specialist: [Name] - [Email] - [Phone]

**Operations Team:**
- DevOps Lead: [Name] - [Email] - [Phone]
- Database Administrator: [Name] - [Email] - [Phone]

### Escalation Matrix

| Issue Severity | Primary Contact | Secondary Contact | Management |
|---------------|-----------------|-------------------|------------|
| Critical | On-call Security | Security Lead | CISO |
| High | Security Team | DevOps Lead | VP Engineering |
| Medium | DevOps Team | Security Team | Engineering Manager |
| Low | Individual Contributor | Team Lead | N/A |

### External Resources

**Vendor Support:**
- Encryption library: [Contact information]
- Key management: [Contact information]
- Monitoring tools: [Contact information]

**Professional Services:**
- Security consultants: [Contact information]
- Compliance auditors: [Contact information]
- Forensic experts: [Contact information]

---

**Document Version:** 1.0
**Last Updated:** 2025-09-17
**Review Date:** 2026-09-17
**Document Owner:** Security Team