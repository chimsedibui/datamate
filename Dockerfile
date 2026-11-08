# Build:  docker build --platform linux/amd64 -t zaic2026_datamate_<team_name>:final .
# Test:   docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
#           -v $PWD/data/public_test:/data:ro -v $PWD/result:/result zaic2026_datamate_<team_name>:final /code/predict.sh
FROM vllm/vllm-openai:v0.10.1.1
ENTRYPOINT []

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1 NCCL_P2P_DISABLE=1

WORKDIR /code

# Organizer-tested fix for the two-GPU NCCL init hang on vLLM 0.10.1.1.
RUN pip install --no-cache-dir --no-deps nvidia-nccl-cu12==2.27.7

# App dependencies go to their own folder so they cannot change vLLM's numpy/torch stack.
COPY requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir --target /code/pydeps -r /code/requirements.txt

# Weights first (large, rarely change) so code edits keep this layer cached.
COPY models/ /code/models/
COPY training_code/ /code/training_code/
COPY predict.sh README.md /code/
COPY src/ /code/src/
RUN chmod +x /code/predict.sh

CMD ["bash", "/code/predict.sh"]
