#!/bin/bash

cleanup() {
    echo "Cleaning up..."
    pkill -P $$ # Kill all child processes of the current script
    exit 0
}

# Trap exit signals and call the cleanup function
trap cleanup SIGINT SIGTERM

# Initialize CUDA before starting Ollama
initialize_cuda() {
    echo "Running CUDA initialization via Python script..."
    python -u cuda_init.py
}

# Run CUDA initialization
initialize_cuda

# Kill any existing ollama processes
pgrep ollama | xargs kill || true

# Start the ollama server and log its output
echo "Starting Ollama server..."
ollama serve 2>&1 | tee ollama.server.log &
OLLAMA_PID=$! # Store the process ID (PID) of the background command

check_server_is_running() {
    echo "Checking if server is running..."
    if cat ollama.server.log | grep -q "Listening"; then
        return 0 # Success
    else
        return 1 # Failure
    fi
}

# Wait for the server to start
while ! check_server_is_running; do
    sleep 5
done
# IF $MODEL_NAME is set, make sure to pull the model, else just skip
if [ -z "$MODEL_NAME" ]; then
    echo "No model name provided. Skipping model pull..."
else
    echo "Pulling model $MODEL_NAME..."
    ollama pull $MODEL_NAME
fi

python -u handler.py $1
