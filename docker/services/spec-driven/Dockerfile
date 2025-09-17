FROM python:3.11-slim
WORKDIR /app
RUN pip install flask
COPY spec_driven_service.py .
EXPOSE 8029
CMD ["python", "spec_driven_service.py"]