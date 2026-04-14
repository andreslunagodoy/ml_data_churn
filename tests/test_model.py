from src.model_loader import load_model

def test_load_model():
    model, preprocessor = load_model()
    assert model is not None, "Model should load"
    assert preprocessor is not None, "Preprocessor should load"

def test_model_has_predict_methods():
    model, _ = load_model()
    assert hasattr(model, "predict"), "Model should have predict method"
    assert hasattr(model, "predict_proba"), "Model should have predict_proba method"

def test_preprocessor_has_transform():
    _, preprocessor = load_model()
    assert hasattr(preprocessor, "transform"), "Preprocessor should have transform method"
