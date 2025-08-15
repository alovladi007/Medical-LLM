# Med-AGI Pilot Go/No-Go Checklist

## Pre-Pilot Requirements

### ✅ Infrastructure
- [ ] **GPU Cluster**: NVIDIA GPUs available with CUDA 12.0+
- [ ] **Kubernetes/Docker**: Container orchestration configured
- [ ] **Load Balancer**: Configured with SSL termination
- [ ] **DNS**: All service endpoints resolvable
- [ ] **Certificates**: Valid SSL certificates installed
- [ ] **Firewall**: Required ports open (8000-8020, 443)

### ✅ Services Health
- [ ] **Imaging Service**: Health check passing
- [ ] **EKG Service**: Health check passing
- [ ] **Eval Service**: Health check passing
- [ ] **Anchor Service**: Health check passing
- [ ] **ModelCards Service**: Health check passing
- [ ] **Ops Service**: Health check passing
- [ ] **Triton Server**: GPU inference operational
- [ ] **Database**: PostgreSQL connected and migrated
- [ ] **Elasticsearch**: Indexed and searchable
- [ ] **MinIO**: S3 storage accessible

### ✅ Security
- [ ] **OIDC Provider**: Configured and tested
- [ ] **JWT Validation**: Token verification working
- [ ] **OPA Policies**: Authorization rules deployed
- [ ] **SIEM Integration**: Logging to security platform
- [ ] **Secrets Management**: All secrets in secure store
- [ ] **Network Policies**: Kubernetes NetworkPolicies applied
- [ ] **PHI Encryption**: Data encrypted at rest and in transit
- [ ] **Audit Logging**: All API calls logged

### ✅ Models & Data
- [ ] **CXR Model**: Deployed to Triton (densenet121_cxr)
- [ ] **CT Model**: Deployed to Triton (resnet50_ct)
- [ ] **MRI Model**: Deployed to Triton (efficientnet_mri)
- [ ] **EKG Model**: Deployed and validated
- [ ] **Model Cards**: Documentation uploaded
- [ ] **Test Data**: Validation dataset available
- [ ] **DICOM Integration**: Connected to PACS (if applicable)

### ✅ Monitoring
- [ ] **Prometheus**: Metrics collection active
- [ ] **Grafana Dashboards**: Deployed and configured
- [ ] **Alert Rules**: Critical alerts configured
- [ ] **PagerDuty**: On-call rotation set up
- [ ] **Log Aggregation**: Centralized logging working
- [ ] **APM**: Application performance monitoring active

### ✅ Testing
- [ ] **Smoke Tests**: All passing (`python3 smoke/smoke.py`)
- [ ] **Load Tests**: K6 tests passing thresholds
- [ ] **Security Scan**: No critical vulnerabilities
- [ ] **Integration Tests**: Cross-service communication verified
- [ ] **Failover Test**: Triton GPU->CPU fallback working
- [ ] **Backup/Restore**: Tested successfully

### ✅ Documentation
- [ ] **API Documentation**: Swagger/OpenAPI available
- [ ] **Runbook**: Operational procedures documented
- [ ] **Architecture Diagram**: Current and approved
- [ ] **Data Flow Diagram**: PHI handling documented
- [ ] **User Guide**: Pilot user documentation ready
- [ ] **Training Materials**: Staff training completed

### ✅ Compliance
- [ ] **HIPAA**: BAA signed, controls implemented
- [ ] **Data Retention**: Policies configured
- [ ] **Consent Management**: Patient consent workflow ready
- [ ] **Access Controls**: RBAC properly configured
- [ ] **Audit Trail**: Compliance logging enabled
- [ ] **Incident Response**: Plan documented and tested

### ✅ Performance
- [ ] **Response Time**: P95 < 2 seconds
- [ ] **Throughput**: >100 requests/second sustained
- [ ] **GPU Utilization**: <80% under normal load
- [ ] **Memory Usage**: <70% on all services
- [ ] **Error Rate**: <0.1% in staging
- [ ] **Availability**: >99.9% over past week

### ✅ Operational Readiness
- [ ] **On-Call Schedule**: 24/7 coverage arranged
- [ ] **Escalation Path**: Defined and communicated
- [ ] **Rollback Plan**: Tested and documented
- [ ] **Communication Plan**: Stakeholders identified
- [ ] **Success Metrics**: KPIs defined and measurable
- [ ] **Feedback Process**: User feedback collection ready

## Go/No-Go Decision Matrix

| Category | Weight | Score (1-5) | Weighted Score |
|----------|--------|-------------|----------------|
| Infrastructure | 20% | ___ | ___ |
| Security | 25% | ___ | ___ |
| Testing | 20% | ___ | ___ |
| Performance | 15% | ___ | ___ |
| Documentation | 10% | ___ | ___ |
| Compliance | 10% | ___ | ___ |
| **TOTAL** | **100%** | | **___** |

**Minimum Score for GO: 4.0**

## Sign-offs

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Engineering Lead | | | |
| Security Officer | | | |
| Compliance Officer | | | |
| Product Owner | | | |
| Medical Director | | | |

## Pilot Launch Decision

- [ ] **GO** - All criteria met, proceed with pilot
- [ ] **NO-GO** - Issues identified, remediation required

### If NO-GO, Issues to Address:
1. ________________________________
2. ________________________________
3. ________________________________

### Target Resolution Date: _______________

## Post-Launch Actions (First 24 Hours)

- [ ] Monitor all service metrics closely
- [ ] Review error logs every 2 hours
- [ ] Check user feedback channels
- [ ] Verify data flow and storage
- [ ] Confirm on-call engineer availability
- [ ] Schedule daily standup for pilot duration
- [ ] Prepare first status report

## Emergency Rollback Criteria

Immediate rollback if any of the following occur:
- [ ] Data breach or security incident
- [ ] Patient safety issue identified
- [ ] System availability <95%
- [ ] Critical data corruption
- [ ] Compliance violation detected

## Notes

_Additional observations or concerns:_

---

**Document Version**: 1.0  
**Last Updated**: [Date]  
**Next Review**: [Date]