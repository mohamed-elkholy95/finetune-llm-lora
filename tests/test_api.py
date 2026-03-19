"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.deploy.api_server import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestModelInfoEndpoint:
    def test_model_info(self):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "lora_config" in data
        assert "supported_models" in data


class TestGenerateEndpoint:
    def test_generate(self):
        resp = client.post("/generate", json={
            "prompt": "Explain machine learning",
            "max_new_tokens": 50,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_text" in data
        assert len(data["generated_text"]) > 0

    def test_generate_empty_prompt(self):
        resp = client.post("/generate", json={"prompt": ""})
        assert resp.status_code == 422


class TestChatEndpoint:
    def test_chat(self):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_new_tokens": 50,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_chat_empty_messages(self):
        resp = client.post("/chat", json={"messages": []})
        assert resp.status_code == 422
