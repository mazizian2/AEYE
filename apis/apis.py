import requests
def customer_search(
        username: str | None = None,
        mobile: str | None = None,
        melicode: str | None = None,
        lastname: str | None = None,
        timeout: float = 10.0
) -> dict:
    url = "https://lte.shabakieh.com/webservice/rest/customer_search"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "login_username": "aeye",
        "login_password": "Dehdar123!@#",
    }

    if username:
        data["username"] = username
    if mobile:
        data["mobile"] = mobile
    if melicode:
        data["melicode"] = melicode
    if lastname:
        data["lastname"] = lastname

    try:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": "request_failed", "detail": str(e)}
    except ValueError:
        return {"error": "invalid_json", "text": resp.text}