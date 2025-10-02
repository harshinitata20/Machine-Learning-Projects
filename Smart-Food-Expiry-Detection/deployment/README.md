# Deployment Instructions for Smart Food Expiry Detection System

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- 10GB free disk space

### 1. Development Setup (Recommended for testing)

```bash
# Clone the repository
git clone <your-repo-url>
cd food-expiry-detection

# Start all services
docker-compose up --build

# Access the applications:
# - Frontend (Streamlit): http://localhost:8501
# - API Documentation: http://localhost:8000/docs
# - API Alternative Docs: http://localhost:8000/redoc
```

### 2. Production Deployment

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d

# Access via Nginx reverse proxy:
# - Frontend: http://localhost
# - API: http://localhost/api
```

## Manual Installation

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv food_expiry_env

# Activate environment
# Windows:
food_expiry_env\Scripts\activate
# Linux/Mac:
source food_expiry_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Backend API

```bash
# Navigate to project root
cd food-expiry-detection

# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python directly:
python -m api.main
```

### 3. Start Frontend

```bash
# In a new terminal, activate environment and run:
streamlit run frontend/app.py --server.port 8501
```

## Cloud Deployment Options

### Option 1: Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry <your-registry> --image food-expiry-api:latest .
az acr build --registry <your-registry> --image food-expiry-frontend:latest -f frontend/Dockerfile .

# Deploy to Container Instances
az container create \
  --resource-group <your-rg> \
  --name food-expiry-api \
  --image <your-registry>.azurecr.io/food-expiry-api:latest \
  --ports 8000
```

### Option 2: AWS ECS with Fargate

```bash
# Push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com

docker tag food-expiry-api:latest <account>.dkr.ecr.us-west-2.amazonaws.com/food-expiry-api:latest
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/food-expiry-api:latest

# Deploy using ECS CLI or AWS Console
```

### Option 3: Google Cloud Run

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/<project-id>/food-expiry-api
gcloud run deploy food-expiry-api \
  --image gcr.io/<project-id>/food-expiry-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 4: Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create food-expiry-detection

# Deploy using Git
git push heroku main

# Or use container deployment
heroku container:push web -a food-expiry-detection
heroku container:release web -a food-expiry-detection
```

## Environment Variables

### Required Environment Variables

```bash
# Database Configuration
DATABASE_URL=sqlite:///./data/food_expiry.db
# or for PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/food_expiry_db

# File Storage
UPLOAD_DIR=./uploads
RESULTS_DIR=./results

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security (for production)
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com

# External Services (optional)
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
SENDGRID_API_KEY=your-sendgrid-key
TELEGRAM_BOT_TOKEN=your-telegram-token

# Model Configuration
YOLO_MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
```

### Setting Environment Variables

#### Windows
```cmd
set DATABASE_URL=sqlite:///./data/food_expiry.db
set API_HOST=0.0.0.0
```

#### Linux/Mac
```bash
export DATABASE_URL=sqlite:///./data/food_expiry.db
export API_HOST=0.0.0.0
```

#### Docker
Create a `.env` file in the project root:
```
DATABASE_URL=sqlite:///./data/food_expiry.db
API_HOST=0.0.0.0
API_PORT=8000
```

## Scaling and Performance

### Horizontal Scaling

1. **Load Balancer Setup**
   - Use Nginx or cloud load balancer
   - Multiple API instances behind load balancer
   - Session affinity not required (stateless API)

2. **Database Optimization**
   - Use PostgreSQL or MySQL for production
   - Implement connection pooling
   - Add database indexes for performance

3. **Caching Strategy**
   - Redis for API response caching
   - CDN for static assets
   - Model result caching

### Monitoring and Logging

1. **Application Monitoring**
   - Prometheus + Grafana setup included
   - Health check endpoints available
   - Custom metrics for food detection accuracy

2. **Log Management**
   - Structured logging with JSON format
   - Log rotation and retention policies
   - Centralized logging with ELK stack (optional)

## Security Considerations

### Production Security Checklist

- [ ] Enable HTTPS with SSL certificates
- [ ] Implement API authentication (JWT tokens)
- [ ] Set up CORS policies appropriately
- [ ] Use environment variables for secrets
- [ ] Regular security updates for dependencies
- [ ] Input validation and sanitization
- [ ] Rate limiting for API endpoints
- [ ] Database connection encryption

### SSL/TLS Setup

```nginx
# Nginx SSL configuration
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://frontend:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port
   netstat -ano | findstr :8000  # Windows
   lsof -ti:8000  # Linux/Mac
   
   # Kill process
   taskkill /PID <PID> /F  # Windows
   kill -9 <PID>  # Linux/Mac
   ```

2. **Permission denied errors**
   ```bash
   # Fix file permissions
   chmod -R 755 /path/to/project
   chown -R $USER:$USER /path/to/project
   ```

3. **Memory issues**
   - Increase Docker memory allocation
   - Use lighter YOLO models (yolov8n instead of yolov8x)
   - Implement model result caching

4. **Database connection issues**
   - Check DATABASE_URL environment variable
   - Ensure database service is running
   - Verify network connectivity between services

### Logs and Debugging

```bash
# View container logs
docker-compose logs api
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f api

# Debug mode
export DEBUG=True
uvicorn api.main:app --reload --log-level debug
```

## Backup and Recovery

### Database Backup

```bash
# SQLite backup
cp data/food_expiry.db backup/food_expiry_$(date +%Y%m%d).db

# PostgreSQL backup
pg_dump -h localhost -U food_expiry_user food_expiry_db > backup/db_$(date +%Y%m%d).sql
```

### Application Backup

```bash
# Backup uploaded files and results
tar -czf backup/uploads_$(date +%Y%m%d).tar.gz uploads/
tar -czf backup/results_$(date +%Y%m%d).tar.gz results/
```

## Performance Tuning

### API Optimization

1. **Enable response caching**
2. **Implement async processing for heavy operations**
3. **Use connection pooling**
4. **Optimize image processing pipeline**

### Frontend Optimization

1. **Enable Streamlit caching**
2. **Optimize chart rendering**
3. **Implement lazy loading**
4. **Compress static assets**

For additional support or questions, refer to the project documentation or create an issue in the GitHub repository.