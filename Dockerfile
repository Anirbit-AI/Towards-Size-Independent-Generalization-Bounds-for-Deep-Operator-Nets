FROM nvcr.io/nvidia/jax:24.04-py3

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN pip install matplotlib tqdm seaborn

# CMD ["bash"]
CMD ["python", "/workspace/main.py"]