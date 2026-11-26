from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_type"] == "LinearSVC"


def test_model_info():
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "Linear SVM" in data["active_classifier"]


def test_predict_real_english():
    res = client.post(
        "/predict",
        json={
            "text": (
                "WASHINGTON (Reuters) - The U.S. Federal Reserve held interest rates steady "
                "on Wednesday while signaling potential adjustments later this year based on "
                "inflation and labor market data."
            )
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "Likely Real"
    assert data["language"] == "en"
    assert data["confidence"] >= 0.62


def test_predict_fake_english():
    res = client.post(
        "/predict",
        json={
            "text": (
                "BREAKING SHOCKING: Secret alien base discovered under the Eiffel Tower and "
                "world leaders have been hiding the truth from citizens for decades!"
            )
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "Likely Fake"
    assert data["language"] == "en"
    assert data["confidence"] >= 0.62


def test_predict_hindi_low_confidence():
    res = client.post(
        "/predict",
        json={
            "text": (
                "नई दिल्ली: भारतीय अंतरिक्ष अनुसंधान संगठन ने आज एक नए संचार उपग्रह के "
                "सफल प्रक्षेपण की घोषणा की।"
            )
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["language"] == "hi"
    assert data["label"] == "Low Confidence"


def test_predict_empty_rejected():
    res = client.post("/predict", json={"text": "   "})
    assert res.status_code == 400


if __name__ == "__main__":
    test_health()
    test_model_info()
    test_predict_real_english()
    test_predict_fake_english()
    test_predict_hindi_low_confidence()
    test_predict_empty_rejected()
    print("All FastAPI + spaCy tests passed!")

