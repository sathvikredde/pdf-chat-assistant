import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
import json
import sqlite3
import os
import io
import base64
import time
from datetime import datetime

# ── Setup ─────────────────────────────────────────────────────────────────────
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
client  = OpenAI(api_key=api_key)

JSON_FILE = "transactions.json"
DB_FILE   = "transactions.db"
TIMEOUT   = 10 * 60  # 10 minutes

# ── DB ────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            uuid              TEXT PRIMARY KEY,
            transaction_date  TEXT,
            type              TEXT,
            parties_involved  TEXT,
            saved_at          TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── JSON / DB helpers ─────────────────────────────────────────────────────────
def load_json():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            return json.load(f)
    return {}

def save_to_json_and_db(transactions: dict):
    existing = load_json()
    existing.update(transactions)
    with open(JSON_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    conn = sqlite3.connect(DB_FILE)
    for uid, data in transactions.items():
        conn.execute(
            "INSERT OR REPLACE INTO transactions VALUES (?,?,?,?,?)",
            (uid,
             data.get("transaction_date", ""),
             data.get("type", ""),
             data.get("parties_involved", ""),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
    conn.commit()
    conn.close()

# ── PDF → text (text layer first, GPT-4o Vision fallback for scanned) ─────────
def pdf_to_page_images_b64(data: bytes) -> list[str]:
    """Convert each PDF page to a base64 PNG string."""
    doc = fitz.open(stream=data, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        images.append(base64.b64encode(pix.tobytes("png")).decode())
    return images

def read_pdf_bytes(data: bytes) -> str:
    """Extract text. If the PDF is scanned, use GPT-4o Vision to read it."""
    doc  = fitz.open(stream=data, filetype="pdf")
    text = "".join(page.get_text() for page in doc).strip()

    if text:
        return text  # normal text-based PDF — done

    # Scanned PDF: send each page image to GPT-4o and ask it to transcribe
    st.info("📷 Scanned PDF detected — using AI Vision to read it (this may take a moment)…")
    page_images = pdf_to_page_images_b64(data)

    all_text = []
    for i, img_b64 in enumerate(page_images):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"This is page {i+1} of a financial document. "
                            "Please transcribe ALL text you see exactly as it appears, "
                            "preserving dates, amounts, names, and transaction details."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }],
            max_tokens=2000,
        )
        all_text.append(resp.choices[0].message.content)

    return "\n\n".join(all_text).strip()

# ── Transaction extractor ─────────────────────────────────────────────────────
EXTRACT_PROMPT = """
From the conversation below, extract every financial transaction that was discussed.
Return ONLY a raw JSON object — no markdown, no backticks, no explanation.

Use this exact format:
{{
  "<generate-a-uuid4-here>": {{
    "transaction_date": "YYYY-MM-DD",
    "type": "credit or debit",
    "parties_involved": "received from X / sent to Y"
  }}
}}

Rules:
- Generate a real UUID4 string for each key (e.g. "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
- Use real dates from the conversation, not "YYYY-MM-DD"
- Use real names/parties from the conversation, not "unknown"
- If no transactions were discussed return exactly: {{}}

Conversation:
{conversation}
"""

def extract_transactions(history: list) -> dict:
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    resp  = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(conversation=convo)}],
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content.strip()
    # strip markdown fences if model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {}

# ── System prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(pdf_text: str) -> str:
    return (
        "You are a helpful financial document assistant. "
        "Answer all questions based ONLY on the PDF content below. "
        "When mentioning transactions always include the exact date, amount, type (credit/debit), and parties involved. "
        "Be concise, accurate, and friendly.\n\n"
        f"--- PDF CONTENT ---\n{pdf_text[:15000]}\n--- END ---"
    )

# ── Session state init ────────────────────────────────────────────────────────
for k, v in {
    "pdf_name":    None,
    "pdf_text":    None,
    "pdf_bytes":   None,
    "history":     [],
    "done":        False,
    "saved":       False,
    "last_active": time.time(),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("📄 PDF Chat Assistant")
st.caption("Upload a PDF, ask questions about it, say **bye** to save and exit.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — upload
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.pdf_text and not st.session_state.done:

    st.info("Browse and select a PDF from your computer.")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded is not None:
        st.session_state.pdf_bytes = uploaded.read()
        st.session_state.pdf_name  = uploaded.name

    if st.session_state.pdf_bytes:
        st.success(f"📎 **{st.session_state.pdf_name}** selected — press the button to begin.")
        if st.button("✅ Load PDF & Start Chat", type="primary"):
            with st.spinner("Reading PDF…"):
                try:
                    text = read_pdf_bytes(st.session_state.pdf_bytes)
                    if not text:
                        st.error("Could not extract any text from this PDF.")
                    else:
                        st.session_state.pdf_text    = text
                        st.session_state.pdf_bytes   = None
                        st.session_state.last_active = time.time()
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — chat
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.done:

    if time.time() - st.session_state.last_active > TIMEOUT and st.session_state.history:
        st.session_state.done = True
        st.rerun()

    st.success(f"✅ Loaded: **{st.session_state.pdf_name}**")

    for msg in st.session_state.history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the PDF… or type 'bye' to finish")

    if user_input:
        st.session_state.last_active = time.time()

        if user_input.strip().lower() in ("bye", "goodbye", "exit", "quit"):
            st.session_state.done = True
            st.rerun()

        st.session_state.history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": build_system_prompt(st.session_state.pdf_text)}
                    ] + st.session_state.history,
                    max_tokens=1000,
                )
            reply = resp.choices[0].message.content
            st.markdown(reply)

        st.session_state.history.append({"role": "assistant", "content": reply})
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — save & download
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.done:

    if not st.session_state.saved:
        with st.spinner("Extracting transactions and saving…"):
            if st.session_state.history:
                txns = extract_transactions(st.session_state.history)
                if txns:
                    save_to_json_and_db(txns)
        st.session_state.saved = True

    saved = load_json()
    st.success("✅ Session complete!")

    if saved:
        st.subheader("Extracted Transactions")
        for uid, data in saved.items():
            with st.expander(f"🧾 {data.get('transaction_date','?')} — {data.get('type','?').capitalize()}"):
                st.write(f"**Parties:** {data.get('parties_involved', '—')}")
                st.caption(f"UUID: {uid}")

        st.download_button(
            label="⬇️ Download transactions.json",
            data=json.dumps(saved, indent=2),
            file_name="transactions.json",
            mime="application/json",
        )
    else:
        st.info("No transactions were found in the conversation.")

    if st.button("🔄 Start New Session"):
        for k in ["pdf_name", "pdf_text", "pdf_bytes", "history", "done", "saved"]:
            st.session_state[k] = [] if k == "history" else None
        st.session_state.done        = False
        st.session_state.saved       = False
        st.session_state.last_active = time.time()
        st.rerun()