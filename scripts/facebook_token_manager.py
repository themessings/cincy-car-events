#!/usr/bin/env python3
"""Facebook token lifecycle management for Graph collectors."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional

import requests
from dateutil import tz

FACEBOOK_GRAPH_VERSION = "v18.0"
TOKEN_STATE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data", "facebook_token_state.json")


@dataclass
class TokenDebugStatus:
    is_valid: bool
    reason: str
    expires_at: Optional[int]
    issued_at: Optional[int]
    scopes: list[str]


class TokenManager:
    def __init__(self, logger: Callable[[str], None]):
        self.log = logger
        self.user_token = (os.getenv("FACEBOOK_ACCESS_TOKEN") or "").strip()
        self.app_id = (os.getenv("FACEBOOK_APP_ID") or "").strip()
        self.app_secret = (os.getenv("FACEBOOK_APP_SECRET") or "").strip()
        self.page_tokens: Dict[str, str] = {}
        self.last_debug: Optional[TokenDebugStatus] = None
        self.refresh_error = ""
        self._load_cached_token()

    def _load_cached_token(self) -> None:
        if self.user_token:
            return
        if not os.path.exists(TOKEN_STATE_PATH):
            return
        try:
            payload = json.loads(open(TOKEN_STATE_PATH, "r", encoding="utf-8").read())
        except Exception:
            return
        cached = (payload.get("facebook_access_token") or "").strip()
        if cached:
            self.user_token = cached
            os.environ["FACEBOOK_ACCESS_TOKEN"] = cached

    def _persist_token(self, token: str, debug: Optional[TokenDebugStatus] = None) -> None:
        os.makedirs(os.path.dirname(TOKEN_STATE_PATH), exist_ok=True)
        payload = {
            "facebook_access_token": token,
            "updated_at_et": datetime.now(tz=tz.gettz("America/New_York")).isoformat(),
            "expires_at": debug.expires_at if debug else None,
            "is_valid": debug.is_valid if debug else None,
        }
        with open(TOKEN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _format_et(self, ts: Optional[int]) -> str:
        if not ts:
            return "unknown"
        return datetime.fromtimestamp(ts, tz=tz.gettz("America/New_York")).isoformat()

    def debug_token(self, token: Optional[str] = None) -> TokenDebugStatus:
        value = (token or self.user_token or "").strip()
        if not value:
            status = TokenDebugStatus(False, "missing", None, None, [])
            self.last_debug = status
            return status

        if not self.app_id or not self.app_secret:
            # Fallback sanity check when app credentials are unavailable.
            try:
                r = requests.get(
                    f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}/me",
                    params={"access_token": value, "fields": "id"},
                    timeout=20,
                )
                if r.status_code == 200:
                    status = TokenDebugStatus(True, "ok_without_debug_token", None, None, [])
                else:
                    status = TokenDebugStatus(False, f"http_{r.status_code}", None, None, [])
            except Exception as ex:
                status = TokenDebugStatus(False, f"debug_failed:{ex}", None, None, [])
            self.last_debug = status
            return status

        app_token = f"{self.app_id}|{self.app_secret}"
        r = requests.get(
            f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}/debug_token",
            params={"input_token": value, "access_token": app_token},
            timeout=20,
        )
        if r.status_code != 200:
            status = TokenDebugStatus(False, f"debug_http_{r.status_code}", None, None, [])
            self.last_debug = status
            return status

        payload = r.json() or {}
        data = payload.get("data") or {}
        is_valid = bool(data.get("is_valid"))
        expires_at = data.get("expires_at")
        issued_at = data.get("issued_at")
        scopes = list(data.get("scopes") or [])
        reason = "ok" if is_valid else "invalid"
        if expires_at and expires_at <= int(time.time()):
            is_valid = False
            reason = "expired"
        status = TokenDebugStatus(is_valid, reason, expires_at, issued_at, scopes)
        self.last_debug = status
        return status

    def exchange_long_lived_user_token(self) -> tuple[bool, str]:
        if not self.user_token:
            return False, "missing_user_token"
        if not self.app_id or not self.app_secret:
            return False, "missing_app_credentials"

        r = requests.get(
            "https://graph.facebook.com/v18.0/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": self.app_id,
                "client_secret": self.app_secret,
                "fb_exchange_token": self.user_token,
            },
            timeout=25,
        )
        if r.status_code != 200:
            return False, f"exchange_http_{r.status_code}:{(r.text or '')[:220]}"

        payload = r.json() or {}
        new_token = (payload.get("access_token") or "").strip()
        if not new_token:
            return False, "exchange_missing_access_token"

        self.user_token = new_token
        os.environ["FACEBOOK_ACCESS_TOKEN"] = new_token
        debug = self.debug_token(new_token)
        self._persist_token(new_token, debug)
        return True, "exchanged"

    def ensure_valid_user_token(self, refresh_days_threshold: int = 7) -> dict:
        status = self.debug_token(self.user_token)
        expires_in_days = None
        if status.expires_at:
            expires_in_days = (status.expires_at - int(time.time())) / 86400.0

        should_refresh = (not status.is_valid) or (
            expires_in_days is not None and expires_in_days < float(refresh_days_threshold)
        )

        refreshed = False
        refresh_reason = "not_needed"
        if should_refresh:
            ok, reason = self.exchange_long_lived_user_token()
            refreshed = ok
            refresh_reason = reason
            if ok:
                status = self.debug_token(self.user_token)
            else:
                self.refresh_error = reason

        self.log(
            "ℹ️ FACEBOOK token status: "
            f"valid={'yes' if status.is_valid else 'no'} "
            f"expires_at_et={self._format_et(status.expires_at)} "
            f"refresh_attempted={'yes' if should_refresh else 'no'} "
            f"refresh_result={'success' if refreshed else refresh_reason}"
        )

        if should_refresh and not refreshed:
            self.log(f"⚠️ Facebook token refresh failed: {refresh_reason}")

        return {
            "valid": "yes" if status.is_valid else "no",
            "reason": status.reason,
            "expires_at": status.expires_at,
            "expires_at_et": self._format_et(status.expires_at),
            "refresh_attempted": should_refresh,
            "refresh_succeeded": refreshed,
            "refresh_error": "" if refreshed else refresh_reason,
        }

    def get_active_user_token(self) -> str:
        return (self.user_token or "").strip()

    def get_page_access_token(self, page_id: str) -> tuple[Optional[str], str]:
        key = (page_id or "").strip()
        if not key:
            return None, "missing_page_id"
        if key in self.page_tokens:
            return self.page_tokens[key], "cached"

        token = self.get_active_user_token()
        if not token:
            return None, "missing_user_token"

        r = requests.get(
            f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}/{key}",
            params={"fields": "access_token", "access_token": token},
            timeout=25,
        )
        if r.status_code != 200:
            return None, f"page_token_http_{r.status_code}:{(r.text or '')[:180]}"

        payload = r.json() or {}
        page_token = (payload.get("access_token") or "").strip()
        if not page_token:
            return None, "page_token_missing"

        self.page_tokens[key] = page_token
        return page_token, "ok"
