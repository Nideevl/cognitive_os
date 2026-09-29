import os
import json
import time
from pathlib import Path
from typing import List, Callable, Any
from openai import OpenAI


STATE_FILE = Path(__file__).resolve().parent.parent / ".key_pointer.json"


class GroqKeyRotator:
    @property
    def total_keys(self) -> int:
        return len(self.keys)
    
    def __init__(
        self,
        api_keys: List[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        state_file: Path = STATE_FILE
    ):
        self.base_url = base_url
        self.state_file = Path(state_file)

        # 1. Load keys from argument or environment variables (GROQ_API_KEY, GROQ_API_KEY_1..N)
        if api_keys:
            self.keys = api_keys
        else:
            self.keys = self._discover_keys()

        if not self.keys:
            raise ValueError("No Groq API keys found in environment or arguments.")

        # 2. Restore pointer from persistent disk state
        self.current_idx = self._load_persisted_index()
        self.clients = [OpenAI(api_key=k, base_url=self.base_url) for k in self.keys]

    def _discover_keys(self) -> List[str]:
        keys = []
        # Check standard single key
        single = os.getenv("GROQ_API_KEY")
        if single:
            keys.append(single.strip())

        # Check numbered keys: GROQ_API_KEY_1 through GROQ_API_KEY_50
        for i in range(1, 51):
            val = os.getenv(f"GROQ_API_KEY_{i}")
            if val and val.strip() not in keys:
                keys.append(val.strip())

        # Check comma-separated pool: GROQ_KEY_POOL="key1,key2,..."
        pool_str = os.getenv("GROQ_KEY_POOL")
        if pool_str:
            for k in pool_str.split(","):
                clean = k.strip()
                if clean and clean not in keys:
                    keys.append(clean)

        return keys

    def _load_persisted_index(self) -> int:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                saved_idx = int(data.get("last_used_index", 0))
                # Bound check against the current key count in case keys were removed
                return saved_idx % len(self.keys)
            except Exception:
                return 0
        return 0

    def _persist_index(self, idx: int):
        """Synchronously persists the active pointer so aborts resume cleanly."""
        try:
            self.state_file.write_text(
                json.dumps({"last_used_index": idx, "updated_at": time.time()}),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"[Warning] Failed to write key pointer to disk: {e}")

    def get_current_client(self) -> OpenAI:
        return self.clients[self.current_idx]

    def rotate(self) -> OpenAI:
        """Rotates to the next key, bounds it with modulo, and flushes to disk."""
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        self._persist_index(self.current_idx)
        print(f"[Key Rotator] Swapped to API Key [{self.current_idx + 1}/{len(self.keys)}]: ...{self.keys[self.current_idx][-6:]}")
        return self.clients[self.current_idx]

    def execute_with_failover(self, call_fn: Callable[[OpenAI], Any], max_retries_per_key: int = 1) -> Any:
        attempts = 0
        total_keys = len(self.keys)
        max_total_attempts = total_keys * (max_retries_per_key + 1)

        while attempts < max_total_attempts:
            client = self.get_current_client()
            try:
                client = self.get_current_client()
                result = call_fn(client)
                # Success: move to the next key so the NEXT call uses a fresh one
                self.current_idx = (self.current_idx + 1) % len(self.keys)
                self._persist_index(self.current_idx)
                return result

            except Exception as e:
                err_msg = str(e).lower()
                # Rate limit (429), quota exhaustion, or server timeout triggers immediate rotation
                if any(x in err_msg for x in ["429", "rate_limit", "quota", "too many requests", "overloaded", "503", "500"]):
                    print(f"\n[Key Alert] Key [{self.current_idx + 1}/{total_keys}] exhausted/throttled: {e}")
                    self.rotate()
                    attempts += 1
                    time.sleep(0.5)
                else:
                    # If it's a programmatic error (not an API quota issue), raise immediately
                    raise e

        raise RuntimeError("All Groq API keys in the rotation pool have been exhausted.")