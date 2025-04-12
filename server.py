import sys
import platform
import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting web automation core server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
