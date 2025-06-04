from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

# Load environment variables dari .env
# load_dotenv()
# api_key = os.getenv("GEMINI_API_TOKEN")
# if not api_key:
#     raise ValueError("API Key untuk Gemini tidak ditemukan. Pastikan sudah diset di file .env dengan key GEMINI_API_TOKEN.")

# # Inisialisasi LLM Gemini
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.0,
#     top_p=1,
#     google_api_key=api_key,    
#     credentials=None,
#     timeout=60
# )


def init_llm(mode: str = "gemini"):
    if mode == "gemini":
        api_key = os.getenv("GEMINI_API_TOKEN")
        if not api_key:
            raise ValueError("API Key GEMINI tidak ditemukan.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            top_p=1,
            google_api_key=api_key,
            timeout=60
        )
    elif mode == "ollama":
        return ChatOllama(model="llama3.1:8b-instruct-q4_K_M")
    else:
        raise ValueError("Mode LLM tidak dikenali. Gunakan 'gemini' atau 'ollama'.")



# ============================ FEW-SHOT EXAMPLES ============================
few_shot_examples = """
Contoh 1:
Pertanyaan: Dalam pembangunan infrastruktur telekomunikasi, bagaimana cara perhitungan persentase TKDN untuk belanja modal atau capital expenditure (Capex) yang digunakan??
```sql
SELECT a.article_number, a.text, r.title
FROM articles a
JOIN regulations r ON a.regulation_id = r.id
WHERE a.text ILIKE '%TKDN%' OR a.text ILIKE '%belanja modal%' OR a.text ILIKE '%capital expenditure%'
LIMIT {top_k};
```

Contoh 2:
Pertanyaan: Apa isi Pasal 10 Peraturan Menteri Komunikasi dan Informatika (PERMENKOMINFO) Nomor 4 Tahun 2016?
```sql
SELECT a.article_number, a.text
FROM articles a
JOIN regulations r ON a.regulation_id = r.id
WHERE 
r.short_type = 'PERMENKOMINFO' AND 
r.number = '4' AND 
r.YEAR = '2016' AND 
a.article_number = '10'
LIMIT {top_k};

Contoh 3: Pasal nomor berapa saja dari PERMENKOMINFO Nomor 26 Tahun 2007 yang sudah tidak berlaku?
```sql  
SELECT
    a.article_number,
    a.title,
    a.status,
    r.title AS regulation_title,
    r.short_type,
    r.number,
    r.year
FROM
    articles a
JOIN
    regulations r ON a.regulation_id = r.id
WHERE
    r.short_type = 'PERMENKOMINFO'
    AND r.number = '26'
    AND r.year = '2007'
    AND a.status = 'ineffective'
```
"""

# ============================ PROMPT TEMPLATE ============================
_postgres_prompt_id = """Kamu adalah seorang ahli SQL untuk sistem hukum Indonesia.

Tugasmu adalah mengubah pertanyaan hukum dari pengguna menjadi query SQL PostgreSQL yang valid dan efisien, berdasarkan struktur database berikut:

{table_info}

Ikuti aturan berikut:
- Gunakan HANYA nama tabel dan kolom yang terdapat di {table_info}.
- Jangan membuat asumsi nama tabel atau kolom yang tidak ada dalam {table_info}.
- Jangan gunakan SELECT *. Ambil hanya kolom yang relevan untuk menjawab pertanyaan.
- Gunakan ILIKE untuk pencocokan teks jika pengguna menanyakan isi pasal atau konten hukum.
- Jika pertanyaan menyebutkan pelaku hukum tertentu seperti "penyelenggara sistem elektronik", "pemerintah", atau "masyarakat", maka pastikan klausa pencarian juga mencakup entitas tersebut menggunakan ILIKE.
- Saat membuat klausa pencarian menggunakan `ILIKE`, gunakan juga padanan kata hukum yang lazim digunakan dalam dokumen peraturan Indonesia.
- Prioritaskan pencarian yang semantik-relevan dan tidak terlalu literal, agar mencakup lebih banyak kemungkinan hasil.
- Untuk pertanyaan yang tidak terlalu spesifik, gabungkan kondisi pencarian menggunakan `OR`** bukan `AND`, agar hasil pencarian lebih luas dan tidak kehilangan konteks penting.
- Jika pertanyaan mengandung singkatan atau akronim, bentuk kueri SQL hanya menggunakan bentuk lengkap tanpa bentuk singkatannya
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
- Jangan tulis ulang pertanyaan pengguna. Jangan tambahkan penjelasan.
- Format akhir HARUS diawali dengan ```sql dan diakhiri dengan ``` seperti ini, tanpa tambahan apa pun:

```sql
    SELECT ...
```
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

def get_sql_chain(llm, mode="zero-shot"):
    prompt = POSTGRES_PROMPT_FEWSHOT_ID if mode == "few-shot" else POSTGRES_PROMPT_ID
    return LLMChain(llm=llm, prompt=prompt)

# Chains
# sql_chain_zero = LLMChain(llm=llm, prompt=POSTGRES_PROMPT_ID)
# sql_chain_fewshot = LLMChain(llm=llm, prompt=POSTGRES_PROMPT_FEWSHOT_ID)

# ============================ GENERATE SQL ============================
def generate_sql(schema: str, question: str, top_k: int = 100, shot_mode: str = "zero-shot", llm_mode: str = "gemini") -> str:
    """
    Generate SQL query dari pertanyaan pengguna.

    Args:
        schema (str): Informasi struktur tabel.
        question (str): Pertanyaan hukum dari user.
        top_k (int): Batas maksimum hasil.
        shot_mode (str): 'zero-shot' atau 'few-shot'
        llm_mode (str): 'gemini' atau 'ollama'
    """
    llm = init_llm(llm_mode)
    chain = get_sql_chain(llm, mode=shot_mode)
    inputs = {
        "input": question,
        "table_info": schema,
        "top_k": top_k
    }
    return chain.run(inputs).strip()