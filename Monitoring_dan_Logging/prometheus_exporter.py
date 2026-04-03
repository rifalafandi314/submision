from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from fastapi.responses import Response
import time
import psutil
import joblib

app = FastAPI()

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# =========================
# METRICS
# =========================
request_count = Counter("request_count", "Total request")
request_success = Counter("request_success", "Total success request")
model_accuracy = Gauge("model_accuracy", "Model accuracy")
cpu_usage = Gauge("cpu_usage_percent", "CPU usage in percent")
memory_usage = Gauge("memory_usage_percent", "Memory usage in percent")
request_latency = Histogram("request_latency_seconds", "Request latency")

# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "ML Monitoring Running"}

@app.get("/predict")
def predict(text: str = "instagram bagus"):
    start_time = time.time()

    request_count.inc()

    # =========================
    # PREDICT REAL MODEL
    # =========================
    text_vec = tfidf.transform([text])
    pred = model.predict(text_vec)[0]

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    prediction = label_map[pred]

    request_success.inc()

    # =========================
    # METRICS UPDATE
    # =========================
    model_accuracy.set(0.91)  # bisa pakai nilai tetap dari training
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)

    latency = time.time() - start_time
    request_latency.observe(latency)

    return {
        "text": text,
        "prediction": prediction
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")