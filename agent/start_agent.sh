# Build the docker image with:
docker build -t openhands_agent  -f agent/Dockerfile .

SANDBOX_ENV_PATH='/usr/local/bin:$PATH'
LLM_API_KEY="AIzaSyD1yNfAGGULAJN3W61ayxlqWkqGYYi1Cxs"
LLM_MODEL="gemini/gemini-2.5-flash-preview-05-20"
#     -e PATH="/home/tommaso/miniconda3/bin:${PATH}" \ PATH="/home/tommaso/miniconda3/bin:${PATH}"
# 
docker run -it \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=openhands_agent \
    -e SANDBOX_USER_ID=$(id -u) \
    -e LLM_API_KEY=$LLM_API_KEY \
    -e LLM_MODEL=$LLM_MODEL \
    -e SANDBOX_ENV_PATH=$SANDBOX_ENV_PATH \
    -e LLM_NATIVE_TOOL_CALLING=true \
    -v ~/.aws:/root/.aws:ro \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$HOME/.openhands-state":/.openhands-state \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app-$(date +%Y%m%d%H%M%S) \
    docker.all-hands.dev/all-hands-ai/openhands:0.39 \
    python -m openhands.cli.main 