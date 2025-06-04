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

1. **JANGAN** membuat atau menebak nama kolom atau nama tabel. Jika kamu menyebutkan kolom yang tidak ada di {table_info}, jawabanmu akan dianggap SALAH. Gunakan hanya kolom yang secara eksplisit ditampilkan.
2. **JANGAN gunakan SELECT \***. Hanya ambil kolom yang relevan.
3. Jika ada lebih dari satu tabel, selalu gunakan alias tabel untuk menghindari ambiguitas (contoh: `a.article_number`, `r.title`).
4. Untuk pencarian isi teks atau konten hukum, gunakan `ILIKE '%kata%'`.
5. Untuk kolom regulations.year yang bertipe angka (integer), seperti `year` gunakan operator `=` saja (bukan ILIKE).
6. Jika kamu tidak yakin nama kolomnya, lebih baik kosongkan atau gunakan hanya kolom yang ada seperti `title`, `text`, `year`, `number`, `article_number`, `name`, atau `status`.
7. Jika pertanyaan mengandung istilah seperti “arti istilah” atau “definisi”, gunakan tabel `definitions`.
8. Untuk isi pasal, kewajiban, hak, sanksi, gunakan tabel `articles` dan JOIN ke `regulations`.
9. Jika menyebutkan jenis peraturan (seperti 'PERMENKOMINFO', 'UU'), filter pakai `short_type`, `number`, dan `year`.
10. Selalu tambahkan `LIMIT {top_k}` di akhir query, kecuali diminta sebaliknya.
11. Jika tidak yakin dengan query, **lebih baik hasilkan query kosong** (`SELECT 'Query tidak dapat dibuat dengan informasi yang tersedia';`) daripada membuat query yang akan error.
12. Format akhir HARUS dibungkus dalam blok ```sql ... ``` tanpa penjelasan tambahan apa pun.

Contoh:
Pertanyaan: Apa isi Pasal 10 dari PERMENKOMINFO Nomor 4 Tahun 2016?
```sql
SELECT a.article_number, a.text
FROM articles a
JOIN regulations r ON a.regulation_id = r.id
WHERE r.short_type = 'PERMENKOMINFO' AND r.number = '4' AND r.year = 2016 AND a.article_number = '10'
LIMIT {top_k};
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

# def fix_ilike_for_integers(sql: str) -> str:
#     """
#     Ganti ILIKE '%angka%' dengan '='.
#     - Kolom integer → tanpa kutip
#     - Kolom teks → dengan kutip
#     """
#     int_cols = ['regulations.year', 'regulations.number', 'articles.chapter_number', 'reglations.amendment']
#     for col in int_cols:
#         pattern = rf"{col}\s+ILIKE\s+'%(\d+)%'"
#         sql = re.sub(pattern, rf"{col} = \1", sql)

#     str_cols = ['articles.article_number']
#     for col in str_cols:
#         pattern = rf"{col}\s+ILIKE\s+'%(\d+)%'"
#         sql = re.sub(pattern, rf"{col} = '\1'", sql)

#     return sql


def remove_invalid_columns(sql: str, valid_columns: list) -> str:
    """
    Hapus kondisi WHERE/JOIN yang menggunakan kolom yang tidak ada dalam skema.
    """
    expr_pattern = re.compile(r"(\b\w+\.\w+\b)\s*(=|ILIKE|>|<|!=)\s*('[^']*'|\d+)", re.IGNORECASE)

    for match in expr_pattern.finditer(sql):
        full_expr = match.group(0)
        col = match.group(1)

        if col not in valid_columns:
            # Hapus seluruh ekspresi dan AND/OR yang terkait
            sql = re.sub(rf"(\s*(AND|OR)\s*)?{re.escape(full_expr)}", "", sql, flags=re.IGNORECASE)

    sql = re.sub(r"\s+", " ", sql).strip()
    sql = re.sub(r"\b(WHERE|AND|OR)\s*($|;)", "", sql, flags=re.IGNORECASE)
    return sql


def generate_sql(schema: str, question: str, top_k: int = 100, shot_mode: str = "zero-shot", llm_mode: str = "gemini") -> str:
    """
    Menghasilkan SQL aman dari LLM, perbaiki ILIKE integer dan bersihkan kolom tak valid.
    """
    llm = init_llm(llm_mode)
    chain = get_sql_chain(llm, mode=shot_mode)

    inputs = {
        "input": question,
        "table_info": schema,
        "top_k": top_k
    }

    response = chain.run(inputs).strip()

    try:
        from query_executor import extract_sql_query_from_response  # Sesuaikan
        raw_sql = extract_sql_query_from_response(response)
        fixed_sql = fix_ilike_for_integers(raw_sql)

        valid_columns = [
            'article_relations.from_article_id', 'article_relations.to_article_id', 
            'article_relations.relation_type', 'articles.id', 'articles.regulation_id', 
            'articles.chapter_number', 'articles.chapter_about', 'articles.article_number', 
            'articles.text', 'articles.status', 'articles.title', 'definitions.id', 
            'definitions.regulation_id', 'definitions.name', 'definitions.definition', 
            'regulation_relations.from_regulation_id', 'regulation_relations.to_regulation_id', 
            'regulation_relations.relation_type', 'regulations.id', 'regulations.url', 
            'regulations.download_link', 'regulations.title', 'regulations.about', 'regulations.type', 
            'regulations.short_type', 'regulations.amendment', 'regulations.number', 'regulations.year', 
            'regulations.institution', 'regulations.issue_place', 'regulations.issue_date', 
            'regulations.effective_date', 'regulations.observation', 'regulations.consideration', 
            'status.id', 'status.repealed', 'status.repeal', 'status.amended', 'status.amend', 'subjects.id', 
            'subjects.subject'
        ]
        cleaned_sql = remove_invalid_columns(fixed_sql, valid_columns)

        return f"```sql\n{cleaned_sql.strip()}\n```"

    except Exception:
        return response
    #return chain.run(inputs).strip()
    
