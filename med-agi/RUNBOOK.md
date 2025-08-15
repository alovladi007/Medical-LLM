# Med-AGI Operational Runbook

## Table of Contents
1. [System Overview](#system-overview)
2. [Deployment Procedures](#deployment-procedures)
3. [Monitoring & Alerts](#monitoring--alerts)
4. [Incident Response](#incident-response)
5. [Rollback Procedures](#rollback-procedures)
6. [Maintenance Tasks](#maintenance-tasks)

## System Overview

### Architecture
- **Microservices**: Imaging, EKG, Eval, Anchor, ModelCards, Ops
- **Infrastructure**: Docker Swarm/Kubernetes, Triton GPU cluster
- **Data Stores**: PostgreSQL, Elasticsearch, MinIO
- **Monitoring**: Prometheus, SIEM integration

### Critical Dependencies
- NVIDIA Triton Server for GPU inference
- OIDC provider for authentication
- OPA for authorization
- External DICOM/PACS systems

## Deployment Procedures

### Pre-deployment Checklist
```bash
# 1. Run smoke tests
python3 smoke/smoke.py --imaging http://staging:8006 ...

# 2. Check database migrations
docker exec med-agi-postgres psql -U medagi -c "\dt"

# 3. Verify model artifacts
ls -la models/

# 4. Check secrets
kubectl get secrets -n med-agi
```

### Production Deployment
```bash
# 1. Tag release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# 2. Build and push images
docker build -t med-agi/imaging:v1.0.0 services/imaging
docker push med-agi/imaging:v1.0.0

# 3. Deploy with rolling update
kubectl set image deployment/imaging imaging=med-agi/imaging:v1.0.0 -n med-agi

# 4. Monitor rollout
kubectl rollout status deployment/imaging -n med-agi

# 5. Run post-deployment tests
./scripts/post_deploy_test.sh
```

## Monitoring & Alerts

### Key Metrics
| Metric | Threshold | Action |
|--------|-----------|--------|
| Error Rate | > 1% | Page on-call |
| P95 Latency | > 2s | Alert team |
| GPU Utilization | > 90% | Scale up |
| Memory Usage | > 80% | Investigate |
| Triton Health | Unhealthy | Failover to CPU |

### Health Check Endpoints
```bash
# Check all services
for port in 8006 8016 8005 8007 8008 8010; do
  curl -s http://localhost:$port/health
done

# Check Triton specifically
curl http://localhost:8006/v1/triton/health
```

### Alert Response

#### High Error Rate
1. Check service logs: `kubectl logs -f deployment/imaging -n med-agi`
2. Review recent deployments
3. Check upstream dependencies
4. Scale horizontally if load-related
5. Rollback if deployment-related

#### Triton Failure
1. Check GPU status: `nvidia-smi`
2. Restart Triton: `kubectl rollout restart deployment/triton`
3. Failover to CPU inference (automatic)
4. Investigate model corruption
5. Re-deploy model repository if needed

## Incident Response

### Severity Levels
- **P1**: Complete service outage, data loss risk
- **P2**: Degraded performance, partial outage
- **P3**: Minor issues, no user impact
- **P4**: Cosmetic issues, improvements

### Response Procedures

#### P1 Incident
1. **Acknowledge** alert within 5 minutes
2. **Assess** impact and affected services
3. **Communicate** status to stakeholders
4. **Mitigate** immediate issues (failover, scale, rollback)
5. **Investigate** root cause
6. **Resolve** underlying issue
7. **Document** incident report

#### Communication Template
```
Subject: [P1] Med-AGI Service Incident

Status: Investigating | Mitigating | Resolved
Impact: [Affected services and users]
Start Time: [ISO timestamp]
Current Actions: [What's being done]
ETA: [Expected resolution time]
```

## Rollback Procedures

### Quick Rollback
```bash
# Kubernetes
kubectl rollout undo deployment/imaging -n med-agi

# Docker Compose
docker-compose --profile prod down
git checkout previous-version
docker-compose --profile prod up -d
```

### Database Rollback
```sql
-- Create restore point before migration
BEGIN;
SAVEPOINT before_migration;

-- If issues occur
ROLLBACK TO SAVEPOINT before_migration;

-- Or restore from backup
pg_restore -U medagi -d medagi backup_20240115.dump
```

### Model Rollback
```bash
# Restore previous model version
aws s3 sync s3://med-agi-models/v1.0.0/ models/ --delete
docker restart med-agi-triton
```

## Maintenance Tasks

### Daily
- [ ] Review error logs and metrics
- [ ] Check disk usage
- [ ] Verify backup completion
- [ ] Review security alerts

### Weekly
- [ ] Update model performance metrics
- [ ] Clean up old logs and artifacts
- [ ] Review and optimize slow queries
- [ ] Security patches assessment

### Monthly
- [ ] Disaster recovery drill
- [ ] Performance benchmarking
- [ ] Capacity planning review
- [ ] Security audit

### Backup Procedures
```bash
# Database backup
pg_dump -U medagi -d medagi > backup_$(date +%Y%m%d).sql

# Model backup
aws s3 sync models/ s3://med-agi-models/backup-$(date +%Y%m%d)/

# Configuration backup
kubectl get all -n med-agi -o yaml > k8s_backup_$(date +%Y%m%d).yaml
```

### Log Rotation
```yaml
# /etc/logrotate.d/med-agi
/var/log/med-agi/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 medagi medagi
    sharedscripts
    postrotate
        docker kill -s USR1 med-agi-*
    endscript
}
```

## Emergency Contacts

| Role | Name | Contact | Escalation |
|------|------|---------|------------|
| On-Call Primary | Rotation | PagerDuty | Immediate |
| On-Call Secondary | Rotation | PagerDuty | 15 min |
| Engineering Lead | TBD | Email/Slack | 30 min |
| Product Owner | TBD | Email | 1 hour |
| Security Team | TBD | security@ | For breaches |

## Useful Commands

```bash
# View all pods
kubectl get pods -n med-agi

# Tail logs
kubectl logs -f deployment/imaging -n med-agi --tail=100

# Execute into container
kubectl exec -it deployment/imaging-xxx -- /bin/bash

# Port forward for debugging
kubectl port-forward service/imaging 8006:8006 -n med-agi

# Check resource usage
kubectl top nodes
kubectl top pods -n med-agi

# Force restart
kubectl rollout restart deployment/imaging -n med-agi

# Scale deployment
kubectl scale deployment/imaging --replicas=3 -n med-agi
```

## Appendix

### Environment Variables
See `.env.example` for complete list

### API Documentation
Swagger UI available at each service's `/docs` endpoint

### Architecture Diagrams
See `docs/architecture.md`

### Security Procedures
See `docs/security.md`