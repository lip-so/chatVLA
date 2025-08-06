#!/bin/bash

echo "Testing Docker build with different configurations..."

# Test minimal build
echo "1. Testing ultra-minimal build (Dockerfile.minimal)..."
docker build -f Dockerfile.minimal -t chatvla-minimal . --progress=plain

if [ $? -eq 0 ]; then
    echo "✅ Minimal build succeeded!"
else
    echo "❌ Minimal build failed"
fi

# Test simple build
echo "2. Testing simple build (Dockerfile.simple)..."
docker build -f Dockerfile.simple -t chatvla-simple . --progress=plain

if [ $? -eq 0 ]; then
    echo "✅ Simple build succeeded!"
else
    echo "❌ Simple build failed"
fi

# Test main build
echo "3. Testing main build (Dockerfile)..."
docker build -f Dockerfile -t chatvla-main . --progress=plain

if [ $? -eq 0 ]; then
    echo "✅ Main build succeeded!"
else
    echo "❌ Main build failed"
fi

echo "Build tests complete!"