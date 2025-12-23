#!/bin/bash

# Navigate to the web-frontend directory
cd "$(dirname "$0")/web-frontend" || exit

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "First time setup: Installing node dependencies..."
    npm install
fi

echo "-------------------------------------------"
echo " Starting LLM Web UI at http://localhost:3000"
echo "-------------------------------------------"

# Start the server using the dev script (ts-node)
npm run dev
