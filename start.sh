#!/bin/bash
# Start both the API server and Streamlit dashboard

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== HEAL Face Recognition System ==="
echo ""

# Check if dependencies are installed
python -c "import insightface" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Initialize DB
echo "Initializing database..."
python scripts/init_db.py

echo ""
echo "Starting API server on http://localhost:8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 3

echo "Starting Dashboard on http://localhost:8501 ..."
PYTHONPATH="$PROJECT_DIR" streamlit run dashboard/app.py --server.port 8501 &
DASHBOARD_PID=$!

echo ""
echo "==================================="
echo "API:       http://localhost:8000"
echo "API Docs:  http://localhost:8000/docs"
echo "Dashboard: http://localhost:8501"
echo "==================================="
echo "Press Ctrl+C to stop all services"

# Wait and clean up on exit
trap "kill $API_PID $DASHBOARD_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
