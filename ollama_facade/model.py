import ollama


def list_models() -> list:
    for key, models in ollama.list():
        return [model for model in models]
    return []


def model_names() -> list[str]:
    models = list(map(lambda x: x.model, list_models()))
    print("models loaded", models)
    return models
