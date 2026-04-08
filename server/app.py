"""
OpenEnv server entry point — re-exports the FastAPI app from the root server module.
"""
import uvicorn
import sys
import os

# Add parent directory to path so we can import the root server module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

def main():
    uvicorn.run("server:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
