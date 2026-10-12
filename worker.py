"""
WORKER MAIN LOOP
================
Script ini jalan terus-menerus (long-running process) di server/VPS.
Tugasnya: polling tabel `projects` di Supabase, cari yang statusnya
'pending', proses satu-satu.

Cara jalankan:
    python worker.py

Untuk production, jalankan sebagai service (systemd) supaya otomatis
restart kalau crash, dan tetap jalan meskipun SSH session ditutup.
"""

import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone

from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, POLL_INTERVAL_SECONDS
from pipeline import (
    download_dataset,
    normalize_and_sentiment,
    run_topic_modeling,
    generate_report,
    upload_results,
)
# Catatan: modul notifications (email) sudah tidak dipakai lagi.
# Status project cukup ditampilkan lewat dashboard (frontend melakukan
# polling/refresh berkala ke tabel `projects`).


# PENTING: pakai SERVICE ROLE KEY di sini, bukan anon key.
# Service role key bypass RLS, karena worker perlu akses semua project,
# bukan cuma milik satu user. JANGAN taruh key ini di kode frontend.
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def claim_next_pending_project():
    """
    Ambil satu project dengan status 'pending', langsung ubah jadi
    'processing' supaya tidak diambil worker lain (kalau nanti kamu
    scale ke lebih dari 1 worker).
    """
    result = (
        supabase.table("projects")
        .select("*")
        .eq("status", "pending")
        .order("created_at", desc=False)  # FIFO: yang paling lama nunggu diproses duluan
        .limit(1)
        .execute()
    )

    if not result.data:
        return None

    project = result.data[0]

    # Update status jadi 'processing' — ini bertindak sebagai "klaim".
    # Catat juga started_at di sini karena ini titik waktu paling akurat
    # untuk "proses mulai".
    started_at = datetime.now(timezone.utc).isoformat()
    supabase.table("projects").update(
        {"status": "processing", "started_at": started_at}
    ).eq("id", project["id"]).execute()

    project["status"] = "processing"
    project["started_at"] = started_at

    # Tabel `projects` tidak menyimpan email — email ada di sistem auth
    # Supabase (auth.users), terpisah dari tabel kita. Karena worker pakai
    # service_role key (akses admin), kita bisa query langsung ke situ.
    try:
        user_response = supabase.auth.admin.get_user_by_id(project["user_id"])
        project["user_email"] = user_response.user.email
    except Exception as e:
        print(f"Gagal ambil email untuk user {project['user_id']}: {e}")
        project["user_email"] = None

    return project


def update_project_status(project_id, status, **extra_fields):
    payload = {"status": status, **extra_fields}
    supabase.table("projects").update(payload).eq("id", project_id).execute()


def process_project(project):
    project_id = project["id"]
    print(f"[{datetime.now()}] Mulai proses project {project_id} ({project['name']})")

    try:
        # 1. Download dataset dari Supabase Storage
        df = download_dataset(supabase, project["original_file_path"])
        update_project_status(project_id, "processing", row_count=len(df))

        # 2. Normalisasi + sentiment classification (LLM, per batch)
        df = normalize_and_sentiment(df)

        # 3. Topic modeling per kategori sentiment
        df = run_topic_modeling(df)

        # 4. Update status: masuk fase generate report.
        # Catat completed_at DI SINI (bukan nanti setelah upload) supaya
        # nilai ini bisa ditampilkan di dalam report itu sendiri.
        completed_at = datetime.now(timezone.utc).isoformat()
        project["completed_at"] = completed_at
        update_project_status(project_id, "generating_report")

        # 5. Generate narrative report (LLM) — sekarang menghasilkan HTML
        # (cadangan/PDF) SEKALIGUS data JSON terstruktur (dipakai halaman
        # Report React di frontend).
        report_html, report_data = generate_report(df, project)

        # 6. Upload hasil (dataset final + report HTML + report JSON + report PDF) ke Storage
        result_file_path, report_path, report_pdf_path, report_json_path = upload_results(
            supabase, project, df, report_html, report_data
        )

        # 7. Tandai selesai
        update_project_status(
            project_id,
            "completed",
            result_file_path=result_file_path,
            report_path=report_path,
            report_pdf_path=report_pdf_path,
            report_json_path=report_json_path,
            completed_at=completed_at,
        )

        # 8. Tidak kirim email lagi — user akan lihat status "completed" dan
        # tombol "Buka Report" / "Download Dataset" langsung di dashboard.
        print(f"[{datetime.now()}] Selesai project {project_id}")

    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        print(f"[{datetime.now()}] ERROR project {project_id}: {error_message}")
        traceback.print_exc()

        update_project_status(project_id, "failed", error_message=error_message)
        # Tidak kirim email — error sudah tersimpan di kolom `error_message`
        # dan ditampilkan langsung di dashboard (lihat app/dashboard/page.tsx).


def main_loop():
    print("Worker started. Polling for pending projects...")
    while True:
        try:
            project = claim_next_pending_project()
            if project:
                process_project(project)
            else:
                time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            # Kalau error terjadi di level polling itu sendiri (misal koneksi
            # Supabase putus), jangan sampai worker mati total — log dan lanjut.
            print(f"[{datetime.now()}] Worker loop error: {e}")
            traceback.print_exc()
            time.sleep(POLL_INTERVAL_SECONDS)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"hydra2go worker ok")

    def log_message(self, *args):
        pass


def start_health_server():
    """Endpoint kecil supaya hosting seperti Hugging Face Spaces menganggap
    container hidup. Hanya aktif kalau env PORT diset (di Spaces: 7860)."""
    port = os.environ.get("PORT")
    if not port:
        return
    server = HTTPServer(("0.0.0.0", int(port)), _HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def run_once():
    """Proses semua project 'pending' lalu keluar. Dipakai di GitHub Actions."""
    print("Worker (once mode): memproses semua project pending...")
    count = 0
    while True:
        project = claim_next_pending_project()
        if not project:
            break
        process_project(project)
        count += 1
    print(f"Selesai. {count} project diproses.")


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        run_once()
    else:
        start_health_server()
        main_loop()