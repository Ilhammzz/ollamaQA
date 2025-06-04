from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
import re
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
r.YEAR = 2016 AND 
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
    AND r.year = 2007
    AND a.status = 'ineffective'
```
"""

# ============================ PROMPT TEMPLATE ============================
_postgres_prompt_id = """Kamu adalah seorang pakar SQL PostgreSQL dan hukum Indonesia. Tugasmu adalah mengubah pertanyaan hukum dari pengguna menjadi SQL query **yang benar-benar valid** dan **bisa langsung dijalankan tanpa error** pada database PostgreSQL.

Perhatikan struktur tabel berikut ini:

{table_info}

Ikuti peraturan ketat berikut:

1. **JANGAN** membuat atau menebak nama kolom atau nama tabel. Hanya gunakan nama kolom dan nama tabel **yang benar-benar ada** di atas ({table_info}).
2. **JANGAN gunakan SELECT \***. Hanya ambil kolom yang relevan.
3. Jika ada lebih dari satu tabel, selalu gunakan alias tabel untuk menghindari ambiguitas (contoh: `a.article_number`, `r.title`).
4. Untuk pencarian isi teks atau konten hukum, gunakan `ILIKE '%kata%'`.
5. Untuk kolom yang bertipe angka (integer), seperti `year` gunakan operator `=` saja (bukan ILIKE).
6. Jika pertanyaan mengandung istilah seperti “arti istilah” atau “definisi”, gunakan tabel `definitions`.
7. Untuk isi pasal, kewajiban, hak, sanksi, gunakan tabel `articles` dan JOIN ke `regulations`.
8. Jika menyebutkan jenis peraturan (seperti 'PERMENKOMINFO', 'UU'), filter pakai `short_type`, `number`, dan `year`.
9. Selalu tambahkan `LIMIT {top_k}` di akhir query, kecuali diminta sebaliknya.
10. Jika tidak yakin dengan query, **lebih baik hasilkan query kosong** (`SELECT 'Query tidak dapat dibuat dengan informasi yang tersedia';`) daripada membuat query yang akan error.
11. Format akhir HARUS dibungkus dalam blok ```sql ... ``` tanpa penjelasan tambahan apa pun.

Contoh:
Pertanyaan: Apa isi Pasal 10 dari PERMENKOMINFO Nomor 4 Tahun 2016?
```sql
SELECT a.article_number, a.text
FROM articles a
JOIN regulations r ON a.regulation_id = r.id
WHERE r.short_type = 'PERMENKOMINFO' AND r.number = '4' AND r.year = 2016 AND a.article_number = '10'
LIMIT {top_k};
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

def fix_ilike_for_integers(sql: str) -> str:
    """
    Mengganti ILIKE '%...' pada kolom integer menjadi operator '='
    """
    int_cols = ['r.year']
    for col in int_cols:
        pattern = rf"{col}\s+ILIKE\s+'%(\d+)%'"
        sql = re.sub(pattern, rf"{col} = \1", sql)
    return sql

def remove_invalid_joins_and_conditions(sql: str, valid_columns: list) -> str:
    """
    Menghapus JOIN dan kondisi WHERE yang menggunakan kolom tidak valid.
    """
    # Hapus JOINs dengan kolom tak valid
    join_pattern = re.compile(r"(JOIN\s+\w+\s+\w+\s+ON\s+[^;]+?)(?=\bJOIN\b|\bWHERE\b|\Z)", re.IGNORECASE | re.DOTALL)
    for match in join_pattern.findall(sql):
        join_cols = re.findall(r"\b\w+\.\w+\b", match)
        if any(col not in valid_columns for col in join_cols):
            sql = sql.replace(match, "")  # remove entire JOIN block

    # Hapus kondisi WHERE/AND/OR dengan kolom tak valid
    condition_pattern = re.compile(r"(\b\w+\.\w+\b)\s*(=|ILIKE|>|<|!=|IN)\s*('[^']*'|\d+)", re.IGNORECASE)
    for match in condition_pattern.findall(sql):
        col = match[0]
        if col not in valid_columns:
            full_expr = f"{col} {match[1]} {match[2]}"
            sql = re.sub(rf"(\bAND\b|\bOR\b)?\s*{re.escape(full_expr)}", "", sql, flags=re.IGNORECASE)

    # Bersihkan spasi/AND/OR berlebih
    sql = re.sub(r"\b(AND|OR)\b\s*(AND|OR)?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql

def extract_sql_block(text: str) -> str:
    """
    Menangkap isi antara ```sql ... ```
    """
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def generate_sql(schema: str, question: str, top_k: int = 100, shot_mode: str = "zero-shot", llm_mode: str = "gemini") -> str:
    """
    Generate SQL query dari pertanyaan pengguna, dengan parsing blok SQL dan validasi sintaks.
    """
    llm = init_llm(llm_mode)
    prompt = POSTGRES_PROMPT_FEWSHOT_ID if shot_mode == "few-shot" else POSTGRES_PROMPT_ID
    chain = LLMChain(llm=llm, prompt=prompt)

    inputs = {
        "input": question,
        "table_info": schema,
        "top_k": top_k
    }

    try:
        raw_output = chain.run(inputs).strip()
    except Exception:
        return "```sql\nSELECT 'Gagal membangkitkan query karena LLM error';\n```"

    extracted_sql = extract_sql_block(raw_output)
    if not extracted_sql:
        return "```sql\nSELECT 'Tidak ditemukan blok ```sql dalam response.';\n```"

    # Perbaikan logika dan validasi
    fixed_sql = fix_ilike_for_integers(extracted_sql)

    valid_cols = [
        'a.article_number', 'a.text', 'a.title', 'a.status', 'a.id', 'a.regulation_id',
        'r.id', 'r.title', 'r.short_type', 'r.type', 'r.number', 'r.year', 'r.status',
        'd.id', 'd.name', 'd.definition', 'd.regulation_id'
    ]
    cleaned_sql = remove_invalid_joins_and_conditions(fixed_sql, valid_cols)

    if not cleaned_sql.strip().lower().startswith("select"):
        return "```sql\nSELECT 'Query gagal dibentuk karena banyak kolom tidak sesuai skema';\n```"

    return f"```sql\n{cleaned_sql.strip()}\n```"