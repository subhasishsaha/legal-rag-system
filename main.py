import requests

API_URL = "http://127.0.0.1:8000/ask"

def ask_api(query: str):
    response = requests.post(
        API_URL,
        json={"query": query}
    )

    # handle errors
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    return response.json()


if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        result = ask_api(query)
        print(f"Answer: {result['answer']}")
        print("-" * 50)