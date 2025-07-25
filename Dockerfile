FROM jamesdolezal/slideflow:latest-tf

# Set working directory
WORKDIR /app

# Copy training script
COPY train_model.py /app/

# Create output directory
RUN mkdir -p /output

# Set environment variables for better GPU utilization
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
ENTRYPOINT ["python", "train_model.py"]
CMD ["--data-dir", "/data", "--output-dir", "/output"]