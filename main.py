# Standard library imports
import os
import json
import logging
import time
from typing import Dict, Any, List

# External dependencies
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

# Local imports
from agent import Agent

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_automation.log")
    ]
)
logger = logging.getLogger("web_automation")

# Load environment variables
load_dotenv()

# Get API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")

# Global state
active_connections = {}
agent = Agent(api_key=API_KEY)

# Create app
app = FastAPI(
    title="Web Automation Core",
    description="Automated web browser with LLM control and user intervention",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def get_index():
    """Serves the main HTML interface."""
    static_html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(static_html_path):
        return FileResponse(static_html_path)
    else:
        logger.error(f"index.html not found in {static_dir}")
        return HTMLResponse("<html><body><h1>Error</h1><p>UI template not found.</p></body></html>", status_code=500)

# --- API Routes ---


@app.get("/api/status")
def get_status():
    """Returns the current agent status"""
    if not agent:
        return JSONResponse(status_code=500, content={"error": "Agent not initialized"})

    status = {
        "is_running": agent.is_running,
        "is_paused": agent.is_paused,
        "current_goal": agent.goal,
        "completed_steps": agent.completed_steps
    }
    return status


@app.get("/api/metrics")
def get_metrics():
    """Returns telemetry metrics for monitoring"""
    if not agent or not hasattr(agent, "llm_service") or not agent.llm_service:
        return JSONResponse(status_code=500, content={"error": "LLM service not initialized"})

    try:
        metrics = agent.llm_service.get_telemetry_report()
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard")
def get_dashboard():
    """Serves the metrics dashboard HTML"""
    metrics_dashboard_path = os.path.join(static_dir, "dashboard.html")
    if os.path.exists(metrics_dashboard_path):
        return FileResponse(metrics_dashboard_path)
    else:
        logger.error(f"dashboard.html not found in {static_dir}")
        return HTMLResponse("<html><body><h1>Error</h1><p>Dashboard template not found.</p></body></html>", status_code=404)

# --- WebSocket Endpoint ---


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = id(websocket)
    active_connections[connection_id] = websocket
    logger.info(f"WebSocket connected. Active: {len(active_connections)}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            command = message.get("command")

            # Handle ping
            if command == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                continue

            # Process commands
            if command == "execute":
                goal = message.get("prompt", "")  # Treat prompt as the goal
                if goal:
                    # Ensure browser is ready
                    if not agent.browser.page or agent.browser.page.is_closed():
                        logger.info(
                            "Browser not initialized, initializing...")
                        init_success = await agent.browser.initialize()
                        if not init_success:
                            await websocket.send_json({"type": "error", "message": "Failed to initialize browser."})
                            continue

                    # Start execution with the goal
                    logger.info(f"Starting execution for goal: {goal}")
                    # Pass goal
                    success = await agent.start_execution(goal, websocket)

                    if not success:
                        logger.error("Agent failed to start execution.")

            elif command == "resume":
                await agent.resume_execution()

            elif command == "stop":
                await agent.stop_execution()

            elif command == "close_browser":
                await agent.browser.close()
                agent.is_running = False  # Ensure state reflects closure
                await websocket.send_json({"type": "browser_closed", "message": "Browser closed"})

            elif command == "get_status":
                status = {
                    "is_running": agent.is_running,
                    "is_paused": agent.is_paused,
                    "goal": agent.goal,
                    "completed_steps_count": len(agent.completed_steps)
                }
                await websocket.send_json({"type": "status_update", "status": status})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up
        if connection_id in active_connections:
            del active_connections[connection_id]
        agent.websocket = None

# On startup, initialize any resources


@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up")

# On shutdown, close resources


@app.on_event("shutdown")
async def shutdown_event():
    await agent.browser.close() if agent else None
    logger.info("Application shutting down, browser closed")


if __name__ == "__main__":
    # Start server
    logger.info("Starting web automation server")
    uvicorn.run(app, host="127.0.0.1", port=8000)
