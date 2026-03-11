import urllib.request
import json


def test_chat():
    payload = json.dumps({"message": "Hello"}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:5001/chat",
        method="POST",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(e.read().decode("utf-8"))


if __name__ == "__main__":
    test_chat()
