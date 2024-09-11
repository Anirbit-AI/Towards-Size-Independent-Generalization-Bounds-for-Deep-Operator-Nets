FROM nvcr.io/nvidia/jax:24.04-py3

WORKDIR /workspace

RUN pip install matplotlib tqdm

CMD ["bash"]