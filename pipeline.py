"""
PIPELINE
========
Kumpulan fungsi untuk tiap tahap pemrosesan data:
1. download_dataset       - ambil file dari Supabase Storage
2. normalize_and_sentiment - LLM call: tulis ulang ejaan + klasifikasi sentiment
3. run_topic_modeling      - BERTopic per kategori sentiment
4. generate_report         - LLM call: susun narasi report
5. upload_results          - simpan hasil balik ke Supabase Storage
"""

import io
import json
import numpy as np
import pandas as pd
import torch
import anthropic
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from weasyprint import HTML

from config import ANTHROPIC_API_KEY, LLM_MODEL, BATCH_SIZE

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ------------------------------------------------------------
# 1. DOWNLOAD DATASET
# ------------------------------------------------------------
def download_dataset(supabase, file_path: str) -> pd.DataFrame:
    file_bytes = supabase.storage.from_("datasets").download(file_path)

    if file_path.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    # Pastikan ada kolom 'id' unik untuk tracking tiap baris sepanjang pipeline
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # TODO: sesuaikan nama kolom teks sesuai format dataset klien kamu.
    # Untuk skeleton ini kita asumsikan kolom teks bernama 'text'.
    if "text" not in df.columns:
        raise ValueError(
            "Kolom 'text' tidak ditemukan di dataset. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

    # Buang baris yang teksnya kurang dari 3 kata — data sependek itu
    # (misal cuma "bagus", "oke sih") tidak cukup informatif untuk
    # normalisasi/sentiment/topic modeling, jadi tidak perlu diproses.
    word_count = df["text"].astype(str).str.split().str.len()
    rows_before = len(df)
    df = df[word_count >= 3]
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"[download_dataset] {rows_dropped} baris dibuang karena kurang dari 3 kata.")

    df = df.reset_index(drop=True)
    return df


# ------------------------------------------------------------
# 2. NORMALISASI + SENTIMENT (digabung 1 LLM call per batch, lebih hemat)
# ------------------------------------------------------------
def _normalize_and_sentiment_batch(items: list[dict]) -> list[dict]:
    prompt = f"""Anda memproses text pelanggan Bahasa Indonesia. Untuk setiap item, lakukan DUA hal:
1. Tulis ulang teks dengan ejaan baku (perbaiki slang, singkatan, typo), TANPA mengubah makna atau nada asli.
2. Klasifikasikan sentimennya: "positive", "negative", atau "neutral".

ATURAN:
- Jangan ubah makna atau intensitas emosi saat menulis ulang
- Sentiment "neutral" untuk text yang informatif/pertanyaan tanpa muatan emosi jelas

Kembalikan HANYA JSON array, tanpa teks lain, format:
[{{"id": "...", "normalized": "...", "sentiment": "..."}}]

Data:
{json.dumps(items, ensure_ascii=False)}
"""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def normalize_and_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        items = [
            {"id": str(row["id"]), "text": row["text"]}
            for _, row in batch.iterrows()
        ]

        try:
            batch_result = _normalize_and_sentiment_batch(items)
            results.extend(batch_result)
        except Exception as e:
            # Kalau satu batch gagal, jangan hentikan seluruh project —
            # catat baris mana yang gagal, biar bisa di-retry manual nanti.
            print(f"Batch {i} gagal: {e}")
            for item in items:
                results.append(
                    {"id": item["id"], "normalized": None, "sentiment": None}
                )

    result_df = pd.DataFrame(results)

    # PENTING: LLM kadang ikut mengembalikan field tambahan (misal 'text')
    # meskipun tidak diminta di prompt. Kalau dibiarkan, ini bentrok dengan
    # kolom 'text' asli saat merge (pandas otomatis rename jadi text_x/text_y).
    # Jadi kita paksa cuma ambil kolom yang memang kita butuhkan.
    result_df = result_df[["id", "normalized", "sentiment"]]

    result_df["id"] = result_df["id"].astype(df["id"].dtype)

    df = df.merge(result_df, on="id", how="left")
    df = df.rename(columns={"normalized": "text_normalized"})

    print(f"[normalize_and_sentiment] Kolom setelah merge: {list(df.columns)}")
    return df


# ------------------------------------------------------------
# 3. TOPIC MODELING per kategori sentiment
# ------------------------------------------------------------
class IndoBERTEmbedder(BaseEmbedder):
    """
    Custom embedding backend untuk BERTopic menggunakan IndoBERT.
    BERTopic secara default hanya support model dari sentence-transformers,
    jadi IndoBERT (model HuggingFace biasa) dibungkus lewat BaseEmbedder
    dengan mean pooling manual.
    """

    def __init__(self, model_name: str = "indobenchmark/indobert-base-p1", device: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def embed(self, documents, verbose: bool = False):
        embeddings = []
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
            pooled = self._mean_pooling(output, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)


def _generate_topic_title(keywords: list[str], sentiment_value: str) -> str:
    """
    Minta Claude membuat judul topic singkat berdasarkan top keywords dari satu cluster BERTopic.
    Fallback ke gabungan keyword kalau panggilan LLM gagal.
    """
    fallback = " / ".join(keywords)

    prompt = f"""Berikut kata kunci utama dari sebuah topik dalam komentar/ulasan pelanggan
berkategori sentimen "{sentiment_value}": {", ".join(keywords)}.

Buatkan SATU judul topik singkat (maksimal 5 kata) dalam Bahasa Indonesia yang jelas
dan mewakili makna kata kunci tersebut. Jawab hanya dengan judulnya saja, tanpa tanda
kutip atau penjelasan tambahan."""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        title = response.content[0].text.strip()
        return title if title else fallback
    except Exception as e:
        print(f"Gagal generate judul topic untuk keywords {keywords}: {e}")
        return fallback


def run_topic_modeling(df: pd.DataFrame) -> pd.DataFrame:
    # Pengecekan defensif: pastikan kolom yang dibutuhkan benar-benar ada
    # sebelum lanjut. Kalau tidak, kasih error yang jelas (bukan KeyError
    # samar), sekaligus print kolom apa saja yang tersedia untuk debugging.
    required_cols = {"text", "text_normalized", "sentiment"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Kolom berikut tidak ditemukan sebelum topic modeling: {missing_cols}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df["topic"] = None
    embedder = IndoBERTEmbedder()

    for sentiment_value in ["positive", "negative", "neutral"]:
        subset = df[df["sentiment"] == sentiment_value]

        # BERTopic butuh jumlah dokumen minimal untuk cluster yang masuk akal.
        # Kalau datanya terlalu sedikit, kasih label default saja.
        if len(subset) < 4:
            df.loc[subset.index, "topic"] = "General"
            continue

        docs = subset["text_normalized"].fillna(subset["text"]).tolist()

        topic_model = BERTopic(
            embedding_model=embedder,  # IndoBERT, dijalankan ulang tiap kelompok sentiment
            min_topic_size=max(3, len(docs) // 20),  # adaptif ke jumlah data
        )
        topics, _ = topic_model.fit_transform(docs)

        # Ambil top keywords tiap cluster, lalu minta Claude buatkan judul topic
        topic_labels = {}
        for topic_id in set(topics):
            if topic_id == -1:
                topic_labels[topic_id] = "Sub Kategori Campuran"
                continue
            keywords = [w for w, _ in topic_model.get_topic(topic_id)[:5]]
            topic_labels[topic_id] = _generate_topic_title(keywords, sentiment_value)

        df.loc[subset.index, "topic"] = [topic_labels[t] for t in topics]

    return df


# ------------------------------------------------------------
# 4. GENERATE NARRATIVE REPORT
# ------------------------------------------------------------
import re
from datetime import datetime, timedelta, timezone as tz


def _clean_text(text: str) -> str:
    """
    Bersihkan artefak Markdown dan tanda baca yang tidak diinginkan dari
    teks hasil LLM. Ini pengaman kedua di luar instruksi prompt, supaya
    walaupun model sesekali lolos menulis format Markdown, hasil akhirnya
    tetap bersih.
    """
    if not text:
        return ""

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\-\*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

    for dash in ["—", "–", "â€”", "â€“"]:
        text = text.replace(dash, ", ")

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _call_llm_json(prompt: str, max_tokens: int = 3000) -> dict:
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


FORMATTING_RULES = """
ATURAN PENULISAN (WAJIB DIIKUTI):
- Tulis kalimat naratif biasa, JANGAN gunakan format Markdown apa pun (tanpa **, tanpa #, tanpa bullet, tanpa numbering manual)
- JANGAN gunakan tanda em dash (—). Gunakan koma atau kalimat baru
- JANGAN menyebutkan angka/persentase spesifik di teksmu — angka akan ditambahkan otomatis oleh sistem terpisah dari data asli
- Bahasa profesional, lugas, untuk dibaca manajemen perusahaan
- Kembalikan HANYA JSON valid, tanpa teks tambahan di luar JSON
"""

# ------------------------------------------------------------
# Tema warna per sentiment
# ------------------------------------------------------------
THEMES = {
    "positive": {"dark": "#173d2c", "light": "#eaf4ee", "accent": "#3d9463", "accent_dark": "#a8dcbb"},
    "negative": {"dark": "#3d1414", "light": "#fbeaea", "accent": "#c2564a", "accent_dark": "#f0a99c"},
    "neutral": {"dark": "#132840", "light": "#e9eef7", "accent": "#5a84b8", "accent_dark": "#a9c3e3"},
}
SENTIMENT_LABELS = {"positive": "Positif", "negative": "Negatif", "neutral": "Netral"}
SENTIMENT_EYEBROW_LAST = {
    "positive": "MENJAGA KEKUATAN",
    "negative": "LANGKAH PERBAIKAN",
    "neutral": "MENGUBAH NETRAL MENJADI POSITIF",
}


# ------------------------------------------------------------
# Helper: metadata (tanggal observasi, waktu proses, catatan)
# ------------------------------------------------------------
def _format_datetime_id(iso_string: str | None, date_only: bool = False) -> str:
    if not iso_string:
        return "-"
    try:
        dt = datetime.fromisoformat(str(iso_string).replace("Z", "+00:00"))
        dt_wib = dt.astimezone(tz(timedelta(hours=7)))
        bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
        if date_only:
            return f"{dt_wib.day:02d} {bulan[dt_wib.month - 1]} {dt_wib.year}"
        return f"{dt_wib.day:02d} {bulan[dt_wib.month - 1]} {dt_wib.year}, {dt_wib.strftime('%H:%M')} WIB"
    except Exception:
        return str(iso_string)


def _detect_observation_dates(df: pd.DataFrame) -> tuple[str, str]:
    """Cari kolom tanggal opsional di dataset (kalau ada) untuk tahu rentang observasi asli."""
    candidate_cols = [c for c in df.columns if c.lower() in
                       ("date", "tanggal", "created_at", "timestamp", "review_date", "waktu")]
    if not candidate_cols:
        return "-", "-"
    try:
        col = candidate_cols[0]
        parsed = pd.to_datetime(df[col], errors="coerce").dropna()
        if parsed.empty:
            return "-", "-"
        return (
            _format_datetime_id(parsed.min().isoformat(), date_only=True),
            _format_datetime_id(parsed.max().isoformat(), date_only=True),
        )
    except Exception:
        return "-", "-"


def _compute_catatan(df: pd.DataFrame) -> str:
    total = len(df)
    failed = df["text_normalized"].isna().sum() if "text_normalized" in df.columns else 0
    if failed > 0:
        pct = round((failed / total) * 100, 1) if total else 0
        return f"{failed} dari {total} data ({pct}%) tidak sepenuhnya berhasil diproses karena kendala teknis saat normalisasi."
    return "Tidak ada catatan khusus. Seluruh data berhasil diproses."


def _pick_sample_quotes(group: pd.DataFrame, n: int = 3) -> list[str]:
    """Ambil sample kutipan dari kolom MENTAH (raw, belum dinormalisasi) 'text'."""
    candidates = group["text"].dropna().astype(str)
    candidates = candidates[candidates.str.len().between(15, 240)]
    if candidates.empty:
        candidates = group["text"].dropna().astype(str)
    return candidates.drop_duplicates().head(n).tolist()


# ------------------------------------------------------------
# Helper: bar chart (untuk halaman overview, versi dark background)
# ------------------------------------------------------------
def _render_bar_rows_html(topic_counts: dict, accent_color: str, unit_label: str = "feedback") -> str:
    if not topic_counts:
        return ""
    sorted_items = sorted(topic_counts.items(), key=lambda x: -x[1])
    max_count = sorted_items[0][1] if sorted_items else 1

    rows = ""
    for label, count in sorted_items:
        pct_width = round((count / max_count) * 100) if max_count else 0
        rows += f"""
        <div class="ov-bar-row">
            <div class="ov-bar-label">{label}</div>
            <div class="ov-bar-track"><div class="ov-bar-fill" style="width:{pct_width}%; background:{accent_color};"></div></div>
            <div class="ov-bar-value">{count}</div>
        </div>"""
    return rows


def _render_stat_card_html(headline_pct: int, headline_text: str, secondary_text: str) -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-pct">{headline_pct}%</div>
        <p class="stat-headline">{headline_text}</p>
        <p class="stat-secondary">{secondary_text}</p>
    </div>
    """


# ------------------------------------------------------------
# LLM: generate konten per section (headline overview + per-topik)
# ------------------------------------------------------------
def _generate_section_content(df: pd.DataFrame, sentiment_value: str) -> dict | None:
    subset = df[df["sentiment"] == sentiment_value]
    if subset.empty:
        return None

    topic_counts = subset["topic"].value_counts().to_dict()
    total_section = len(subset)

    grouped_text = ""
    for topic, group in subset.groupby("topic"):
        sample_texts = group["text_normalized"].dropna().head(15).tolist()
        grouped_text += f"\n\nTopik: {topic} ({len(group)} feedback)\n"
        grouped_text += "\n".join(f"- {t}" for t in sample_texts)

    label = SENTIMENT_LABELS[sentiment_value]
    prompt = f"""Anda adalah analis riset yang menganalisis feedback pelanggan berkategori sentiment {label}.

Berikut data feedback, dikelompokkan per sub kategori/topik:
{grouped_text}

{FORMATTING_RULES}

Kembalikan JSON dengan struktur persis seperti ini:
{{
  "headline_text": "1 kalimat highlight tentang pola utama/topik paling dominan dalam kategori {label} ini (TANPA angka)",
  "secondary_text": "1-2 kalimat tentang topik yang lebih jarang muncul atau catatan tambahan (TANPA angka)",
  "topics": [
    {{
      "topic_label": "harus PERSIS sama dengan label topik di data",
      "description": "2-3 kalimat menjelaskan pola/insight dalam topik ini",
      "recommendation": "1-2 kalimat rekomendasi tindakan konkret terkait topik ini"
    }}
  ]
}}
"""
    try:
        parsed = _call_llm_json(prompt, max_tokens=3000)
    except Exception as e:
        print(f"Gagal generate section content '{sentiment_value}': {e}")
        parsed = {"headline_text": "", "secondary_text": "", "topics": []}

    parsed["headline_text"] = _clean_text(parsed.get("headline_text", ""))
    parsed["secondary_text"] = _clean_text(parsed.get("secondary_text", ""))
    for t in parsed.get("topics", []):
        t["description"] = _clean_text(t.get("description", ""))
        t["recommendation"] = _clean_text(t.get("recommendation", ""))
        t["count"] = topic_counts.get(t["topic_label"], 0)
        t["pct"] = round((t["count"] / total_section) * 100) if total_section else 0
        t["samples"] = _pick_sample_quotes(subset[subset["topic"] == t["topic_label"]])

    parsed["topics"].sort(key=lambda t: -t["count"])
    parsed["topic_counts"] = topic_counts
    parsed["total_section"] = total_section

    # Hitung headline_pct dari top-1 atau top-2 topik (mengikuti pola referensi)
    sorted_counts = sorted(topic_counts.values(), reverse=True)
    top_sum = sum(sorted_counts[:2]) if len(sorted_counts) >= 2 else sum(sorted_counts[:1])
    parsed["headline_pct"] = round((top_sum / total_section) * 100) if total_section else 0

    return parsed


# ------------------------------------------------------------
# Render halaman-halaman untuk satu section sentiment
# ------------------------------------------------------------
def _render_sentiment_pages(section_data: dict | None, sentiment_value: str, total_all: int) -> str:
    theme = THEMES[sentiment_value]
    label = SENTIMENT_LABELS[sentiment_value]

    if not section_data:
        return f'<div class="page" style="background:{theme["light"]};"><h1>Tidak ada data {label}</h1></div>'

    pages_html = ""

    # --- Halaman overview (dark bg) ---
    bar_rows = _render_bar_rows_html(section_data["topic_counts"], theme["accent_dark"])
    stat_card = _render_stat_card_html(
        section_data["headline_pct"], section_data["headline_text"], section_data["secondary_text"]
    )
    overview_page = f"""
    <div class="page theme-dark" style="background:{theme['dark']};">
        <div class="eyebrow" style="color:{theme['accent_dark']};">{label.upper()} &middot; GAMBARAN UMUM</div>
        <h1>Ringkasan Feedback {label}</h1>
        <p class="muted">Jumlah feedback per sub kategori (total {section_data['total_section']} feedback {label.lower()})</p>
        <div class="two-col">
            <div class="col-left">{bar_rows}</div>
            <div class="col-right">{stat_card}</div>
        </div>
        {_page_footer()}
    </div>
    """
    pages_html += overview_page

    # --- Halaman per sub kategori (light bg) ---
    n_topics = len(section_data["topics"])
    for i, t in enumerate(section_data["topics"], start=1):
        quotes_html = "".join(
            f'<div class="quote-card"><p>&ldquo;{q}&rdquo;</p></div>' for q in t["samples"]
        ) or '<div class="quote-card"><p>Tidak ada sampel data.</p></div>'

        detail_page = f"""
        <div class="page" style="background:{theme['light']};">
            <div class="eyebrow" style="color:{theme['accent']};">{label.upper()} &middot; SUB KATEGORI {i} DARI {n_topics}</div>
            <div class="two-col">
                <div class="col-left">
                    <h1>{t['topic_label']}</h1>
                    <p class="big-number" style="color:{theme['accent']};">{t['count']}
                        <span class="big-number-suffix">feedback ({t['pct']}% dari {label.lower()})</span>
                    </p>
                    <p>{t['description']}</p>
                </div>
                <div class="col-right">
                    <div class="eyebrow" style="color:{theme['accent']};">SAMPEL FEEDBACK</div>
                    {quotes_html}
                </div>
            </div>
            {_page_footer()}
        </div>
        """
        pages_html += detail_page

    # --- Halaman rekomendasi (dark bg) ---
    rec_cards = ""
    for i, t in enumerate(section_data["topics"], start=1):
        rec_cards += f"""
        <div class="rec-card">
            <div class="rec-priority" style="color:{theme['accent_dark']};">PRIORITAS {i} &middot; {t['count']} FEEDBACK</div>
            <div class="rec-title">{t['topic_label']}</div>
            <p>{t['recommendation']}</p>
        </div>
        """
    rec_page = f"""
    <div class="page theme-dark" style="background:{theme['dark']};">
        <div class="eyebrow" style="color:{theme['accent_dark']};">{label.upper()} &middot; {SENTIMENT_EYEBROW_LAST[sentiment_value]}</div>
        <h1>Rekomendasi</h1>
        {rec_cards}
        {_page_footer()}
    </div>
    """
    pages_html += rec_page

    return pages_html


def _page_footer() -> str:
    return '<div class="page-footer"><span class="footer-title"></span><span class="footer-page"></span></div>'


# ------------------------------------------------------------
# Sankey diagram 3 level: Total -> Sentiment -> Sub Kategori
# ------------------------------------------------------------
def _sankey_level(items: list[tuple], x_left: float, node_w: float, x_right: float,
                   y0: float, y1: float, gap: float) -> tuple[list[str], list[tuple]]:
    total = sum(v for _, v, _ in items) or 1
    height = y1 - y0
    avail = height - gap * (len(items) - 1) if len(items) > 1 else height
    avail = max(avail, 1)

    left_cursor = y0
    right_cursor = y0
    parts = []
    right_nodes = []
    mid_x = (x_left + node_w + x_right) / 2

    for label, count, color in items:
        lh = (count / total) * height
        ly0, ly1 = left_cursor, left_cursor + lh
        left_cursor += lh

        rh = (count / total) * avail
        ry0, ry1 = right_cursor, right_cursor + rh
        right_cursor += rh + gap

        path = (
            f'M {x_left + node_w},{ly0:.1f} '
            f'C {mid_x:.1f},{ly0:.1f} {mid_x:.1f},{ry0:.1f} {x_right},{ry0:.1f} '
            f'L {x_right},{ry1:.1f} '
            f'C {mid_x:.1f},{ry1:.1f} {mid_x:.1f},{ly1:.1f} {x_left + node_w},{ly1:.1f} Z'
        )
        parts.append(f'<path d="{path}" fill="{color}" opacity="0.4" />')
        parts.append(f'<rect x="{x_right}" y="{ry0:.1f}" width="{node_w}" height="{(ry1 - ry0):.1f}" fill="{color}" rx="2" />')
        right_nodes.append((label, color, ry0, ry1, count))

    return parts, right_nodes


def _render_full_sankey_svg(df: pd.DataFrame, height: int = 460) -> str:
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    total = len(df)
    order = ["positive", "negative", "neutral"]
    sent_items = [(SENTIMENT_LABELS[s], sentiment_counts.get(s, 0), THEMES[s]["accent"])
                  for s in order if sentiment_counts.get(s, 0) > 0]
    if not sent_items or total == 0:
        return ""

    # Kumpulkan semua label sub kategori dulu untuk menghitung lebar yang
    # dibutuhkan supaya teks TIDAK terpotong, apapun panjang labelnya.
    all_topic_labels = []
    for sent_key in order:
        sub = df[df["sentiment"] == sent_key]
        for t_label, t_count in sub["topic"].value_counts().items():
            all_topic_labels.append(f"{t_label} \u00b7 {t_count}")
    longest_label_chars = max((len(s) for s in all_topic_labels), default=20)
    # Perkiraan lebar per karakter untuk font-size 12px sans-serif
    label_area_width = max(260, int(longest_label_chars * 6.6))

    x0, node_w = 10, 22
    x1 = 190
    x2 = 420
    width = x2 + node_w + 8 + label_area_width + 30
    gap1 = 22

    parts = [f'<svg viewBox="0 0 {width} {height + 10}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">']

    # Node "Total" di kolom paling kiri
    parts.append(f'<rect x="{x0}" y="0" width="{node_w}" height="{height}" fill="#333" rx="3" />')
    parts.append(f'<text x="{x0}" y="{height + 22}" font-size="13" fill="#333" font-weight="bold">Total &middot; {total}</text>')

    level1_parts, sent_nodes = _sankey_level(sent_items, x0, node_w, x1, 0, height, gap1)
    parts.extend(level1_parts)

    for label, color, ry0, ry1, count in sent_nodes:
        mid_y = (ry0 + ry1) / 2 + 5
        parts.append(f'<text x="{x1 + node_w / 2}" y="{mid_y:.1f}" font-size="13" fill="white" font-weight="bold" text-anchor="middle">{label} &middot; {count}</text>')

    # Level 2: per-sentiment -> sub kategori
    for sent_key, (label, color, ry0, ry1, count) in zip(order, sent_nodes):
        sub = df[df["sentiment"] == sent_key]
        topic_counts = sub["topic"].value_counts().to_dict()
        topic_items = sorted(topic_counts.items(), key=lambda x: -x[1])
        topic_items = [(t, c, color) for t, c in topic_items]
        if not topic_items:
            continue
        level2_parts, topic_nodes = _sankey_level(topic_items, x1, node_w, x2, ry0, ry1, gap=6)
        parts.extend(level2_parts)
        for t_label, t_color, t_ry0, t_ry1, t_count in topic_nodes:
            mid_y = (t_ry0 + t_ry1) / 2 + 4
            parts.append(f'<text x="{x2 + node_w + 8}" y="{mid_y:.1f}" font-size="12" fill="#333">{t_label} &middot; {t_count}</text>')

    parts.append('</svg>')
    return "".join(parts)


# ------------------------------------------------------------
# Net Sentiment Score (NSS) & Reputasi
# ------------------------------------------------------------
def _compute_nss(sentiment_counts: dict) -> dict:
    total = sum(sentiment_counts.values()) or 1
    pct_pos = round((sentiment_counts.get("positive", 0) / total) * 100)
    pct_neg = round((sentiment_counts.get("negative", 0) / total) * 100)
    pct_neu = round((sentiment_counts.get("neutral", 0) / total) * 100)
    nss = pct_pos - pct_neg

    if nss >= 50:
        reputasi = "Sangat Baik"
    elif nss >= 20:
        reputasi = "Baik"
    elif nss >= 0:
        reputasi = "Cukup"
    elif nss >= -30:
        reputasi = "Perlu Perhatian"
    else:
        reputasi = "Kritis"

    return {"nss": nss, "pct_pos": pct_pos, "pct_neg": pct_neg, "pct_neu": pct_neu, "reputasi": reputasi}


def _render_reputation_gauge_html(nss: int) -> str:
    marker_pct = ((nss + 100) / 200) * 100
    color = "#3d9463" if nss >= 20 else ("#c2564a" if nss < 0 else "#b8860b")
    return f"""
    <div class="gauge-card">
        <div class="eyebrow">NILAI REPUTASI</div>
        <div class="gauge-value" style="color:{color};">{'+' if nss > 0 else ''}{nss}</div>
        <div class="gauge-track">
            <div class="gauge-marker" style="left:{marker_pct}%;"></div>
        </div>
        <div class="gauge-scale"><span>-100</span><span>0</span><span>+100</span></div>
    </div>
    """


# ------------------------------------------------------------
# CSS keseluruhan
# ------------------------------------------------------------
REPORT_CSS = """
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; color: #222; }

@page { size: A4 landscape; margin: 0; }

.page { padding: 55px 65px; page-break-after: always; position: relative; min-height: 100%; }
.page:last-child { page-break-after: auto; }
.page.theme-dark { color: #f2f2f2; }
.page.theme-dark h1, .page.theme-dark p, .page.theme-dark .stat-secondary { color: #f2f2f2; }

@media screen {
    body { background: #ddd; margin: 0; padding: 30px 0 90px 0; }
    .page { display: none; width: 1100px; min-height: 640px; margin: 0 auto 20px auto;
            border-radius: 6px; box-shadow: 0 2px 14px rgba(0,0,0,0.15); }
    .page.active { display: block; }
    .nav-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #1a1a1a; color: white;
               display: flex; align-items: center; justify-content: center; gap: 20px; padding: 14px; z-index: 999; }
    .nav-bar button { background: white; color: #1a1a1a; border: none; padding: 8px 18px; border-radius: 4px;
                       font-weight: bold; cursor: pointer; font-size: 0.9rem; }
    .nav-bar button:disabled { opacity: 0.35; cursor: not-allowed; }
    .nav-bar span { font-size: 0.9rem; }
}
@media print { .nav-bar { display: none !important; } }

.eyebrow { text-transform: uppercase; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px; color: #666; margin-bottom: 8px; }
h1 { font-family: Georgia, 'Times New Roman', serif; font-size: 2.1rem; margin: 0 0 14px 0; }
p { line-height: 1.55; margin: 8px 0; }
.muted { color: #777; font-size: 0.9rem; }

table.meta { border-collapse: collapse; margin-top: 20px; width: 100%; }
table.meta td { padding: 10px 0; border-bottom: 1px solid #e5e5e5; font-size: 0.95rem; }
table.meta td:first-child { color: #666; width: 320px; }
table.meta td:last-child { font-weight: 600; }

.two-col { display: flex; gap: 40px; align-items: flex-start; margin-top: 18px; }
.two-col .col-left { flex: 1.1; }
.two-col .col-right { flex: 1; }
.two-col.split-30-70 .col-left { flex: 0 0 30%; }
.two-col.split-30-70 .col-right { flex: 1; }

.ov-bar-row { display: flex; align-items: center; margin-bottom: 14px; font-size: 0.9rem; }
.ov-bar-label { width: 170px; flex-shrink: 0; padding-right: 10px; }
.ov-bar-track { flex-grow: 1; background: rgba(255,255,255,0.12); border-radius: 4px; height: 20px; overflow: hidden; }
.theme-dark .ov-bar-track { background: rgba(255,255,255,0.12); }
.page:not(.theme-dark) .ov-bar-track { background: rgba(0,0,0,0.08); }
.ov-bar-fill { height: 100%; border-radius: 4px; }
.ov-bar-value { width: 50px; text-align: right; font-weight: bold; padding-left: 10px; }

.stat-card { background: rgba(255,255,255,0.08); border-radius: 10px; padding: 26px; }
.stat-pct { font-family: Georgia, serif; font-size: 3rem; font-weight: bold; line-height: 1; margin-bottom: 12px; }
.stat-headline { font-weight: 600; }
.stat-secondary { opacity: 0.8; font-size: 0.9rem; }

.big-number { font-family: Georgia, serif; font-size: 3.2rem; font-weight: bold; margin: 10px 0; line-height: 1; }
.big-number-suffix { font-family: 'Segoe UI', Arial, sans-serif; font-size: 1rem; font-weight: normal; color: #555; margin-left: 8px; }

.quote-card { background: rgba(255,255,255,0.6); border: 1px solid rgba(0,0,0,0.08); border-radius: 8px; padding: 16px 18px; margin-bottom: 12px; }
.quote-card p { margin: 0; font-style: italic; }

.rec-card { background: rgba(255,255,255,0.08); border-radius: 8px; padding: 16px 20px; margin-bottom: 14px; }
.rec-priority { font-size: 0.75rem; font-weight: bold; letter-spacing: 0.5px; margin-bottom: 4px; }
.rec-title { font-family: Georgia, serif; font-weight: bold; font-size: 1.1rem; margin-bottom: 6px; }
.rec-card p { margin: 0; }

.gauge-card { background: #f2f0ea; border-radius: 10px; padding: 26px; max-width: 320px; }
.gauge-value { font-family: Georgia, serif; font-size: 3rem; font-weight: bold; margin: 10px 0; }
.gauge-track { position: relative; height: 8px; border-radius: 4px; margin: 16px 0 6px 0;
               background: linear-gradient(to right, #c2564a, #b8860b, #3d9463); }
.gauge-marker { position: absolute; top: -4px; width: 4px; height: 16px; background: #222; border-radius: 2px; transform: translateX(-2px); }
.gauge-scale { display: flex; justify-content: space-between; font-size: 0.8rem; color: #777; }

.page-footer { position: absolute; bottom: 20px; left: 65px; right: 65px; font-size: 0.75rem; color: #999;
               display: flex; justify-content: space-between; }
"""

NAV_SCRIPT = """
<div class="nav-bar">
    <button id="btn-prev" onclick="hydraGoToPage(-1)">&larr; Previous</button>
    <span id="page-indicator"></span>
    <button id="btn-next" onclick="hydraGoToPage(1)">Next &rarr;</button>
</div>
<script>
    let hydraCurrentPage = 0;
    const hydraPages = document.querySelectorAll('.page');

    function hydraRenderPage() {
        hydraPages.forEach((p, i) => p.classList.toggle('active', i === hydraCurrentPage));
        document.getElementById('page-indicator').innerText =
            'Halaman ' + (hydraCurrentPage + 1) + ' dari ' + hydraPages.length;
        document.getElementById('btn-prev').disabled = hydraCurrentPage === 0;
        document.getElementById('btn-next').disabled = hydraCurrentPage === hydraPages.length - 1;
    }

    function hydraGoToPage(delta) {
        const target = hydraCurrentPage + delta;
        if (target >= 0 && target < hydraPages.length) {
            hydraCurrentPage = target;
            hydraRenderPage();
        }
    }

    // Navigasi pakai tombol panah kiri/kanan di keyboard, supaya
    // report ini enak dipakai baik dibuka langsung maupun di dalam
    // iframe/modal di dashboard.
    document.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowRight') hydraGoToPage(1);
        if (e.key === 'ArrowLeft') hydraGoToPage(-1);
    });

    hydraRenderPage();
</script>
"""


def generate_report(df: pd.DataFrame, project: dict) -> str:
    total = len(df)
    sentiment_counts = df["sentiment"].value_counts().to_dict()

    # Prioritaskan tanggal observasi yang diisi user manual di form.
    # Kalau tidak diisi, coba deteksi otomatis dari kolom tanggal di dataset (kalau ada).
    if project.get("observation_start_date") or project.get("observation_end_date"):
        obs_start = _format_datetime_id(project.get("observation_start_date"), date_only=True) if project.get("observation_start_date") else "-"
        obs_end = _format_datetime_id(project.get("observation_end_date"), date_only=True) if project.get("observation_end_date") else "-"
    else:
        obs_start, obs_end = _detect_observation_dates(df)
    proc_start = _format_datetime_id(project.get("started_at"))
    proc_end = _format_datetime_id(project.get("completed_at"))
    # Catatan dari user (diisi saat create project) diprioritaskan.
    # Kalau user tidak mengisi apa-apa, tampilkan catatan otomatis
    # (misal info kalau ada data yang gagal diproses).
    user_notes = (project.get("notes") or "").strip()
    catatan = user_notes if user_notes else _compute_catatan(df)

    cover_page = f"""
    <div class="page" style="background:#f7f5f0;">
        <h1>{project['name']}</h1>
        <table class="meta">
            <tr><td>Tanggal Observasi Dimulai</td><td>{obs_start}</td></tr>
            <tr><td>Tanggal Observasi Selesai</td><td>{obs_end}</td></tr>
            <tr><td>Waktu Data Processing Dimulai</td><td>{proc_start}</td></tr>
            <tr><td>Waktu Data Processing Selesai</td><td>{proc_end}</td></tr>
            <tr><td>Jumlah Data</td><td>{total} feedback ({sentiment_counts.get('positive', 0)} positif &middot; {sentiment_counts.get('negative', 0)} negatif &middot; {sentiment_counts.get('neutral', 0)} netral)</td></tr>
            <tr><td>Catatan</td><td>{catatan}</td></tr>
        </table>
        {_page_footer()}
    </div>
    """

    positive_data = _generate_section_content(df, "positive")
    negative_data = _generate_section_content(df, "negative")
    neutral_data = _generate_section_content(df, "neutral")

    positive_pages = _render_sentiment_pages(positive_data, "positive", total)
    negative_pages = _render_sentiment_pages(negative_data, "negative", total)
    neutral_pages = _render_sentiment_pages(neutral_data, "neutral", total)

    sankey_svg = _render_full_sankey_svg(df)
    sankey_page = f"""
    <div class="page" style="background:#f7f5f0;">
        <div class="eyebrow">KESIMPULAN &middot; RINGKASAN AKHIR</div>
        <h1>Alur Feedback per Sub Kategori</h1>
        {sankey_svg}
        {_page_footer()}
    </div>
    """

    nss_data = _compute_nss(sentiment_counts)
    gauge_html = _render_reputation_gauge_html(nss_data["nss"])

    conclusion_prompt = f"""Susun paragraf kesimpulan untuk laporan analisis customer feedback.

Data: total {total} feedback. {nss_data['pct_pos']}% positif, {nss_data['pct_neg']}% negatif, {nss_data['pct_neu']}% netral.
Net Sentiment Score: {nss_data['nss']} (kategori reputasi: {nss_data['reputasi']}).

{FORMATTING_RULES}
Kecuali untuk angka-angka di atas (boleh disebutkan ulang persis), jangan menambah angka lain yang tidak ada di data.

Kembalikan JSON: {{"kesimpulan": "1 paragraf (4-6 kalimat) yang merangkum temuan utama dan rekomendasi prioritas ke depan"}}
"""
    try:
        concl = _call_llm_json(conclusion_prompt, max_tokens=600)
        kesimpulan_text = _clean_text(concl.get("kesimpulan", ""))
    except Exception as e:
        print(f"Gagal generate kesimpulan: {e}")
        kesimpulan_text = (
            f"Dari {total} feedback, {nss_data['pct_pos']}% positif, {nss_data['pct_neg']}% negatif, "
            f"dan {nss_data['pct_neu']}% netral, dengan Net Sentiment Score {nss_data['nss']} "
            f"({nss_data['reputasi']})."
        )

    reputasi_page = f"""
    <div class="page" style="background:#f7f5f0;">
        <div class="eyebrow">KESIMPULAN &middot; RINGKASAN AKHIR</div>
        <h1>Kesimpulan</h1>
        <div class="two-col split-30-70">
            <div class="col-left">{gauge_html}<p style="margin-top:10px;font-weight:600;">Reputasi: {nss_data['reputasi']}</p></div>
            <div class="col-right"><p>{kesimpulan_text}</p></div>
        </div>
        {_page_footer()}
    </div>
    """

    body = cover_page + positive_pages + negative_pages + neutral_pages + sankey_page + reputasi_page
    # Tandai halaman pertama sebagai aktif (untuk mode screen/interaktif)
    body = body.replace('<div class="page"', '<div class="page active"', 1)

    report_html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>{project['name']}</title>
<style>{REPORT_CSS}</style>
</head>
<body>
{body}
{NAV_SCRIPT}
</body>
</html>
"""

    # ------------------------------------------------------------
    # Data terstruktur (JSON) — dipakai halaman Report React di frontend,
    # supaya tampilan report bisa interaktif (tab per-sentiment dll),
    # bukan cuma render HTML statis.
    # ------------------------------------------------------------
    def _section_to_json(data: dict | None):
        if not data:
            return {"total": 0, "topics": []}
        return {
            "total": data["total_section"],
            "headline_pct": data["headline_pct"],
            "headline_text": data["headline_text"],
            "secondary_text": data["secondary_text"],
            "topics": [
                {
                    "name": t["topic_label"],
                    "count": t["count"],
                    "pct": t["pct"],
                    "description": t["description"],
                    "recommendation": t["recommendation"],
                    "samples": t["samples"],
                }
                for t in data["topics"]
            ],
        }

    report_data = {
        "project_name": project["name"],
        "file_name": (project.get("original_file_path") or "").split("/")[-1],
        "records": total,
        "observation_start": obs_start,
        "observation_end": obs_end,
        "processing_start": proc_start,
        "processing_end": proc_end,
        "notes": catatan,
        "counts": {
            "positive": sentiment_counts.get("positive", 0),
            "negative": sentiment_counts.get("negative", 0),
            "neutral": sentiment_counts.get("neutral", 0),
        },
        "nss": nss_data["nss"],
        "reputasi": nss_data["reputasi"],
        "pct_pos": nss_data["pct_pos"],
        "pct_neg": nss_data["pct_neg"],
        "pct_neu": nss_data["pct_neu"],
        "kesimpulan": kesimpulan_text,
        "sections": {
            "positive": _section_to_json(positive_data),
            "negative": _section_to_json(negative_data),
            "neutral": _section_to_json(neutral_data),
        },
    }

    return report_html, report_data


# ------------------------------------------------------------
# 5. UPLOAD HASIL
# ------------------------------------------------------------
def upload_results(supabase, project: dict, df: pd.DataFrame, report_html: str, report_data: dict):
    user_id = project["user_id"]
    project_id = project["id"]

    result_path = f"{user_id}/{project_id}/result.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    supabase.storage.from_("datasets").upload(
        result_path, csv_bytes, {"content-type": "text/csv; charset=utf-8", "upsert": "true"}
    )

    report_path = f"{user_id}/{project_id}/report.html"
    supabase.storage.from_("datasets").upload(
        report_path,
        report_html.encode("utf-8"),
        {"content-type": "text/html; charset=utf-8", "upsert": "true"},
    )

    # report.json — data terstruktur yang dipakai halaman Report React
    # di frontend (bukan cuma buka file HTML statis).
    report_json_path = f"{user_id}/{project_id}/report.json"
    supabase.storage.from_("datasets").upload(
        report_json_path,
        json.dumps(report_data, ensure_ascii=False).encode("utf-8"),
        {"content-type": "application/json; charset=utf-8", "upsert": "true"},
    )

    report_pdf_path = f"{user_id}/{project_id}/report.pdf"
    try:
        pdf_bytes = HTML(string=report_html).write_pdf()
        supabase.storage.from_("datasets").upload(
            report_pdf_path,
            pdf_bytes,
            {"content-type": "application/pdf", "upsert": "true"},
        )
    except Exception as e:
        print(f"Gagal generate PDF untuk project {project_id}: {e}")
        report_pdf_path = None

    return result_path, report_path, report_pdf_path, report_json_path