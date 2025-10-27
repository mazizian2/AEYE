import requests
def customer_search(info:dict[str,str],
        # username: str | None = None,
        # mobile: str | None = None,
        # melicode: str | None = None,
        # lastname: str | None = None,
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

    if info["username"]:
        data["username"] = info["username"]
    if info["mobile"]:
        data["mobile"] = info["mobile"]
    if info["melicode"]:
        data["melicode"] = info["melicode"]
    if info["lastname"]:
        data["lastname"] =  info["lastname"]

    try:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": "request_failed", "detail": str(e)}
    except ValueError:
        return {"error": "invalid_json", "text": resp.text}