"""
Quick helper: list the models available on your DO Serverless Inference account,
so we use the EXACT model IDs (not guesses) in the comparison.

Run:  python src/list_models.py
"""

from dotenv import load_dotenv

from inference import make_client

load_dotenv()  # read .env so DO_INFERENCE_KEY is available


def main():
    client = make_client()
    models = client.models.list()
    rows = sorted(m.id for m in models.data)
    print(f"{len(rows)} models available:\n")
    for model_id in rows:
        print(f"  {model_id}")


if __name__ == "__main__":
    main()
