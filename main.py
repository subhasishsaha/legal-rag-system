import requests

API_URL = "https://legal-rag-system-production-6df2.up.railway.app/ask"

def ask_api(query: str):
    response = requests.post(
        API_URL,
        json={"query": query}
    )

    # handle errors
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    return response.json()


def health_check():
    response = requests.get("https://legal-rag-system-production-6df2.up.railway.app/health")
    if response.status_code != 200:
        raise Exception(f"Health Check Failed: {response.status_code} - {response.text}")
    return response.json()


if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        result = ask_api(query)
        print(f"Answer: {result['answer']}")
        print("-" * 50)
