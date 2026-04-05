from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
import streamlit as st


def check_health(server_url: str) -> dict | None:
    try:
        resp = httpx.get(f"{server_url}/health", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def get_ocr_providers(server_url: str) -> list[str]:
    try:
        resp = httpx.get(f"{server_url}/api/v1/cheque/available-ocr-providers", timeout=5.0)
        resp.raise_for_status()
        return resp.json().get("ocr_providers", [])
    except Exception:
        return []


def extract_cheque(
    server_url: str,
    image_bytes: bytes,
    filename: str,
    prompt: str,
    output_format: str,
    ocr_provider: str,
    use_gpu: bool,
) -> dict:
    with httpx.Client(timeout=120.0) as client:
        files = {"image": (filename, image_bytes, "image/jpeg")}
        data = {
            "prompt": prompt,
            "output_format": output_format,
            "ocr_provider": ocr_provider,
            "use_gpu": use_gpu,
        }
        response = client.post(
            f"{server_url}/api/v1/cheque/extract",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return response.json()


def main():
    st.set_page_config(page_title="Cheque OCR Extractor", page_icon="🏦", layout="wide")

    st.title("🏦 Cheque OCR Extractor")

    with st.sidebar:
        st.header("Configuration")

        server_url = st.text_input(
            "Server URL",
            value="http://localhost:8000",
        )

        health = check_health(server_url)
        if health:
            st.success(f"🟢 Connected — {health['ocr_provider']} / {health['llm_provider']}")
        else:
            st.error("🔴 Server unreachable")
            st.stop()

        st.divider()

        ocr_providers = get_ocr_providers(server_url)
        selected_ocr = st.selectbox(
            "OCR Provider",
            options=ocr_providers,
            index=0,
        )

        use_gpu = st.checkbox("Use GPU (if available)", value=False)

    st.subheader("Upload Cheque Image")
    uploaded_file = st.file_uploader(
        "Choose a cheque image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Cheque", use_container_width=True)

    prompt = st.text_area(
        "Prompt",
        placeholder="Extract all available fields from this cheque.",
        value="Extract all available fields from this cheque.",
        height=100,
    )

    output_format = st.radio(
        "Output Format",
        options=["json", "markdown"],
        horizontal=True,
    )

    if st.button("Extract", type="primary", disabled=not uploaded_file):
        with st.spinner("Processing cheque..."):
            try:
                result = extract_cheque(
                    server_url=server_url,
                    image_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    prompt=prompt,
                    output_format=output_format,
                    ocr_provider=selected_ocr,
                    use_gpu=use_gpu,
                )

                if result.get("error"):
                    st.error(f"Error: {result['error']}")

                if output_format == "json":
                    st.subheader("Results")
                    st.json(result.get("extracted_data", {}))
                else:
                    markdown_text = result.get("markdown", "")
                    if not markdown_text and not result.get("error"):
                        data = result.get("extracted_data", {})
                        lines = ["# Cheque Extraction Results\n"]
                        rows = []
                        for key, value in data.items():
                            label = key.replace("_", " ").title()
                            if isinstance(value, dict):
                                lines.append(f"### {label}\n")
                                for k, v in value.items():
                                    rows.append((k.replace("_", " ").title(), v if v is not None else "N/A"))
                            else:
                                rows.append((label, value if value is not None else "N/A"))
                        if rows:
                            lines.append("| Field | Value |")
                            lines.append("|-------|-------|")
                            for label, value in rows:
                                lines.append(f"| {label} | {value} |")
                            lines.append("")
                        if result.get("ocr_confidence") is not None:
                            lines.append(f"**OCR Confidence:** {result['ocr_confidence']:.1f}%\n")
                        markdown_text = "\n".join(lines)

                    st.subheader("Results")
                    st.markdown(markdown_text)

                    st.download_button(
                        label="📥 Download Markdown",
                        data=markdown_text,
                        file_name="cheque_extraction.md",
                        mime="text/markdown",
                    )

            except httpx.ConnectError:
                st.error(f"Could not connect to server at {server_url}")
            except httpx.HTTPStatusError as e:
                st.error(f"Server error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
