from src.model_loader import load_model

def test_load_model():
    model, preprocessor = load_model()
    assert model is not None, "Model should load"
    assert preprocessor is not None, "Preprocessor should load"