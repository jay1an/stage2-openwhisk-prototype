from dataclasses import dataclass
from typing import Any, Dict

import requests
from requests.auth import HTTPBasicAuth
from urllib3.exceptions import InsecureRequestWarning


@dataclass
class OpenWhiskClient:
    apihost: str
    auth: str
    namespace: str = "guest"
    verify_tls: bool = False
    timeout_sec: int = 60

    def __post_init__(self) -> None:
        if ":" not in self.auth:
            raise ValueError("OpenWhisk auth must use the format UUID:SECRET")
        self.uuid, self.secret = self.auth.split(":", 1)
        self.apihost = self.apihost.rstrip("/")
        if not self.verify_tls:
            requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

    def invoke_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = (
            f"{self.apihost}/api/v1/namespaces/{self.namespace}/actions/"
            f"{action}?blocking=true&result=true"
        )
        resp = requests.post(
            url,
            json=params,
            auth=HTTPBasicAuth(self.uuid, self.secret),
            verify=self.verify_tls,
            timeout=self.timeout_sec,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenWhisk invoke failed: status={resp.status_code}, body={resp.text[:500]}"
            )
        return resp.json()
