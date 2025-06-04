from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOllama(model="llama3:8b-instruct-q4_K_M")


# ============================ FEW-SHOT EXAMPLES ============================
few_shot_examples = """
Contoh 1:
Pertanyaan: Dalam pembangunan infrastruktur telekomunikasi, bagaimana cara perhitungan persentase TKDN untuk belanja modal atau capital expenditure (Capex) yang digunakan??
SQLQuery:
SELECT a.article_number, a.text, r.title
FROM articles a
JOIN regulations r ON a.regulation_id = r.id
WHERE a.text ILIKE '%TKDN%' OR a.text ILIKE '%belanja modal%' OR a.text ILIKE '%capital expenditure%';
LIMIT {top_k};

Contoh 2:
Pertanyaan: Apa isi Pasal 10 Peraturan Menteri Komunikasi dan Informatika (PERMENKOMINFO) Nomor 4 Tahun 2016??
SQLQuery:
SELECT a.article_number, a.text
FROM articles a
JOIN regulations r ON a.regulation_id = r.id
WHERE 
r.short_type = 'PERMENKOMINFO' AND 
r.number = '4' AND 
r.YEAR = '2016' AND 
a.article_number = '10'
LIMIT {top_k};
"""

# ============================ PROMPT TEMPLATE ============================
_postgres_prompt_id = """Kamu adalah seorang ahli SQL untuk sistem hukum Indonesia.

Tugasmu adalah mengubah pertanyaan hukum dari pengguna menjadi query SQL PostgreSQL yang valid dan efisien, berdasarkan struktur database berikut:

{table_info}

Ikuti aturan berikut:
- Gunakan hanya nama tabel dan kolom yang terdapat di `{table_info}`.
- Bungkus semua nama kolom dengan tanda kutip ganda ("), contoh: "text", "title".
- Jangan gunakan SELECT *. Ambil hanya kolom yang relevan untuk menjawab pertanyaan.
- Gunakan ILIKE untuk pencocokan teks jika pengguna menanyakan isi pasal atau konten hukum.
- Jika pertanyaan menyebutkan pelaku hukum tertentu seperti "penyelenggara sistem elektronik", "pemerintah", atau "masyarakat", maka pastikan klausa pencarian juga mencakup entitas tersebut menggunakan ILIKE.
- Saat membuat klausa pencarian menggunakan `ILIKE`, gunakan padanan kata hukum yang lazim digunakan dalam dokumen peraturan Indonesia.
- Prioritaskan pencarian yang semantik-relevan dan tidak terlalu literal, agar mencakup lebih banyak kemungkinan hasil.
- Untuk pertanyaan yang tidak terlalu spesifik, gabungkan kondisi pencarian menggunakan `OR`** bukan `AND`, agar hasil pencarian lebih luas dan tidak kehilangan konteks penting.
- Jika pertanyaan mengandung singkatan atau akronim, bentuk kueri SQL hanya menggunakan bentuk lengkap tanpa bentuk singkatannya.
- Jika pertanyaan berkaitan dengan definisi istilah, gunakan tabel "definitions".
- Gunakan tabel "definitions" hanya jika pertanyaan merujuk pada istilah hukum formal yang memiliki definisi eksplisit, seperti: "Apa arti", "Apa definisi", atau jika konteks menunjukkan bahwa istilah tersebut memang biasa didefinisikan secara langsung dalam hukum.
- Namun jika pertanyaan mengandung frasa konseptual yang bukan istilah baku, seperti "pemetaan urusan pemerintahan daerah", carilah di tabel "articles" yang memuat isi peraturan atau penjelasan administratif.
- Jika pertanyaan berkaitan dengan isi pasal, kewajiban, hak, atau sanksi, gunakan tabel "articles", dan JOIN ke "regulations" untuk mendapatkan nama regulasi.
- Semua regulasi dalam database ini sudah terbatas pada bidang teknologi informasi, jadi tidak perlu filter seperti `kategori = 'teknologi informasi'`.
- Jika pertanyaan menyebutkan jenis regulasi seperti 'Undang-Undang (UU)', 'Peraturan Pemerintah (PP)', 'PERMENKOMINFO', dll, maka kamu HARUS menyertakan filter berdasarkan kolom `short_type`, `number`, dan `year`.
- Pada tabel 'regulations', type adalah jenis peraturan, dan short_type adalah singkatan dari jenis peraturan.
- Tabel `articles` punya kolom `status` yang nilainya langsung `'effective'` atau `'ineffective'` yang dapat digunakan untuk mengetahui status peraturan tersebut (masih berlaku/tidak berlaku).
- Tabel 'regulations relations' memiliki kolom `relation_type` yang menunjukkan hubungan antar peraturan, yaitu 'mengubah', dan 'diubah oleh'. Gunakan ini untuk pertanyaan yang berkaitan dengan perubahan peraturan.
- Jika pertanyaan berkaitan dengan status peraturan, ambil juga kolom 'status' nya.
- Gunakan tabel article_relations jika pengguna menanyakan apakah sebuah pasal diamandemen.
- Selalu gunakan LIMIT {top_k} untuk membatasi jumlah hasil, kecuali jika diminta lain oleh pengguna.
- Jangan gunakan format markdown seperti ```sql atau tanda kode lainnya.
- Jangan tulis ulang pertanyaan pengguna. Jangan tambahkan penjelasan.
- Format akhir HARUS seperti ini, tanpa tambahan apa pun:

SQLQuery:
SELECT ...
"""

PROMPT_SUFFIX_ID = """Gunakan hanya tabel berikut:
{table_info}

Pertanyaan: {input}
"""

# Zero-shot prompt
POSTGRES_PROMPT_ID = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_postgres_prompt_id + PROMPT_SUFFIX_ID,
)

# Few-shot prompt
POSTGRES_PROMPT_FEWSHOT_ID = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=few_shot_examples + _postgres_prompt_id + PROMPT_SUFFIX_ID,
)

# Chains
sql_chain_zero = LLMChain(llm=llm, prompt=POSTGRES_PROMPT_ID)
sql_chain_fewshot = LLMChain(llm=llm, prompt=POSTGRES_PROMPT_FEWSHOT_ID)

# ============================ GENERATE SQL ============================
def generate_sql(schema: str, question: str, top_k: int = 5, mode: str = "zero-shot") -> str:
    """
    Generate SQL dari pertanyaan bahasa alami menggunakan schema database dan batasan top_k.
    Pilih mode: "zero-shot" atau "few-shot"
    """
    inputs = {
        "input": question,
        "table_info": schema,
        "top_k": top_k
    }
    if mode == "few-shot":
        return sql_chain_fewshot.run(inputs).strip()
    return sql_chain_zero.run(inputs).strip()
