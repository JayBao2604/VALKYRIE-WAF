import requests

WAF_URL = "http://192.168.48.137:80/"
session = requests.Session()


def test_against_waf(payload: str) -> bool:
    """
    Send payload using GET to WAF endpoint with 'payload' as query param.
    Return True if blocked (e.g., status != 200), False if bypassed.
    """
    try:
        response = session.get(
            WAF_URL,
            params={"payload": payload},
            timeout=5,
        )
        print(f"[DEBUG] Payload: {payload} → Status Code: {response.status_code}")
        if response.status_code == 200:
            return 1  # bypassed → label = 1
        else:
            return 0  # blocked (403, 400, etc.) → label = 0

    except Exception as e:
        # Nếu lỗi kết nối, coi như bị chặn
        return True
