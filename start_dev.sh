#!/bin/bash
set -e

echo "==================================================="
echo " Starting Akshara OCR Development Environment"
echo "==================================================="

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker could not be found. Please install Docker Desktop."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "ERROR: npm could not be found. Please install Node.js."
    exit 1
fi

# Stop any existing running containers
echo "Stopping any existing orphaned containers..."
docker-compose down

# Start Backend Services
echo "Starting Backend Services (API, Postgres, Redis, Celery, Model)..."
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Wait for services to initialize
echo "Waiting for services to initialize..."
sleep 5

# Show status of backend containers
echo ""
echo "--- Backend Service Status ---"
docker-compose ps
echo "------------------------------"
echo ""

# Start Frontend Dev Server
echo "Starting Frontend Development Server..."
if [ -d "frontend" ]; then
    cd frontend
    
    # Check if node_modules exists, if not, install dependencies
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    
    # Run dev server
    echo "Frontend server starting. Access the application at: http://localhost:5173"
    npm run dev
else
    echo "ERROR: Frontend directory not found!"
    exit 1
fi
