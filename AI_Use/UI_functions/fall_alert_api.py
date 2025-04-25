# fall_alert_api.py
import requests

BASE_URL = "https://port-0-autoreportsystem-back-m8u790x9772c113e.sel4.cloudtype.app"

def get_id_by_device(device_id: int) -> int:
    url = f"{BASE_URL}/api/userstatus/{device_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("id")
    except requests.RequestException as e:
        print("Error getting user ID:", e)
        return None

def send_status_update(id: int, status: int) -> bool:
    url = f"{BASE_URL}/api/userstatus/status"
    payload = {
        "id": id,
        "status": status
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print("Success:", response.json())
        return True
    except requests.RequestException as e:
        print("Error sending status update:", e)
        return False