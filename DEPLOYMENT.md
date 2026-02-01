# Deployment Guide
## From Local to Production at Scale

**Last Updated**: 2026-01-31

---

## Deployment Levels

### Level 1: Local Development (Jupyter)
**Use Case**: Experimentation, learning
**Cost**: $0
**Setup Time**: Immediate

```bash
jupyter lab
# Open notebook and run experiments
```

---

### Level 2: Local API (Docker)
**Use Case**: Development, testing
**Cost**: $0
**Setup Time**: 10 minutes

```bash
# Build
docker build -t transformer-vljepa:local .

# Run
docker run -p 8000:8000 transformer-vljepa:local

# Test
curl http://localhost:8000/health
```

---

### Level 3: Single Server (AWS EC2, DigitalOcean)
**Use Case**: MVP, demos, small traffic
**Cost**: $50-100/month
**Capacity**: ~100 req/day

#### AWS EC2 Deployment
```bash
# 1. Launch EC2 instance (t3.medium)
# 2. SSH into instance
ssh -i key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 3. Install Docker
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker $USER

# 4. Pull and run
docker pull yourusername/transformer-vljepa:latest
docker run -d -p 8000:8000 --name api transformer-vljepa:latest

# 5. Setup HTTPS with Let's Encrypt
sudo apt install -y nginx certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### Level 4: Auto-Scaling Cloud (AWS ECS, GCP Cloud Run)
**Use Case**: Growing product
**Cost**: $200-500/month
**Capacity**: 10K-100K req/day

#### Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/transformer-vljepa

# Deploy with auto-scaling
gcloud run deploy transformer-vljepa \
  --image gcr.io/PROJECT_ID/transformer-vljepa \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 1 \
  --max-instances 10 \
  --cpu 2 \
  --memory 4Gi
```

---

### Level 5: Kubernetes (EKS, GKE, AKS)
**Use Case**: Enterprise, high availability
**Cost**: $1000+/month
**Capacity**: 1M+ req/day

#### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformer-vljepa
spec:
  replicas: 3
  selector:
    matchLabels:
      app: transformer-vljepa
  template:
    metadata:
      labels:
        app: transformer-vljepa
    spec:
      containers:
      - name: api
        image: yourusername/transformer-vljepa:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: transformer-vljepa-service
spec:
  selector:
    app: transformer-vljepa
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

```bash
kubectl apply -f deployment.yaml
kubectl get services  # Get external IP
```

---

## Monitoring Setup

### Prometheus + Grafana

```bash
# Install Prometheus
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Install Grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana

# Access Grafana
kubectl port-forward service/grafana 3000:80
# Open http://localhost:3000
```

---

## CI/CD Pipeline

### Automated Deployment
```yaml
# .github/workflows/deploy.yml
# On push to main → test → build → deploy
```

See AGENTS.md for complete pipeline.

---

## HTTPS & Security

### Let's Encrypt
```bash
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

### Security Checklist
- [ ] HTTPS enabled
- [ ] API authentication (JWT)
- [ ] Rate limiting configured
- [ ] Secrets in environment variables
- [ ] Firewall configured
- [ ] DDoS protection (Cloudflare)

---

For detailed deployment instructions, see Phase 7 stories in SPRINT_STORIES.md.

Last Updated: 2026-01-31
