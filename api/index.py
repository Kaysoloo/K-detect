"""
Vercel serverless entry point for K-Detect.
Vercel auto-detects Python from api/index.py + requirements.txt.
"""
from app.main import app
