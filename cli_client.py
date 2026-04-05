from __future__ import annotations

import sys
from pathlib import Path

import httpx

from config.settings import settings


def call_server(
    image_path: str,
    prompt: str,
    output_format: str,
    server_url: str | None = None,
) -> dict:
    url = server_url or settings.server_url

    file_path = Path(image_path)
    if not file_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, "rb") as f:
        files = {"image": (file_path.name, f, "image/jpeg")}
        data = {"prompt": prompt, "output_format": output_format}

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{url}/api/v1/cheque/extract",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            print(
                f"Error: Could not connect to server at {url}. "
                f"Is the server running? Start it with: uvicorn api.app:app --host 0.0.0.0 --port {settings.server_port}",
                file=sys.stderr,
            )
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Error: Server returned HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
            sys.exit(1)
