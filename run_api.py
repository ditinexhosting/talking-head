#!/usr/bin/env python3
# coding: utf-8

"""
Entry point for the LivePortrait Flask API.

Usage:
    python run_api.py                        # default: 0.0.0.0:5000
    python run_api.py --port 8080
    python run_api.py --preload              # load models before first request
"""

import argparse
from api.app import app, get_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LivePortrait Flask API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load models at startup instead of on first request",
    )
    cfg = parser.parse_args()

    if cfg.preload:
        get_pipeline()

    app.run(host=cfg.host, port=cfg.port, debug=False, threaded=True)
