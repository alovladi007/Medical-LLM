#!/usr/bin/env python3
"""
Simple demo server for Med-AGI System
Serves the HTML demo page
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Configuration
PORT = 8080
DIRECTORY = Path(__file__).parent

class DemoHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers for demo
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Med-AGI Demo Server                        ║
╚══════════════════════════════════════════════════════════════╝

Starting demo server...
Server running at: http://localhost:{PORT}

To view the demo:
1. Open your browser
2. Navigate to: http://localhost:{PORT}

Press Ctrl+C to stop the server
""")
    
    try:
        with socketserver.TCPServer(("", PORT), DemoHTTPRequestHandler) as httpd:
            print(f"✓ Server started successfully on port {PORT}")
            print("─" * 60)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down demo server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()