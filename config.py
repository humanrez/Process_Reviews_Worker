"""
CONFIG
======
Semua nilai sensitif diambil dari environment variable, JANGAN
hardcode langsung di kode. Buat file .env di folder yang sama
(lihat .env.example), lalu load pakai python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")  # opsional, email sudah tidak dipakai
EMAIL_FROM = os.environ.get("EMAIL_FROM", "ptsatukliksaja@gmail.com")

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS") or "10")
LLM_MODEL = os.environ.get("LLM_MODEL") or "claude-sonnet-4-6"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE") or "15")
