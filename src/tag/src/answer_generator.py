from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

# Inisialisasi LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.0,
#     top_p=1,
#     google_api_key=os.getenv("GEMINI_API_TOKEN"),
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
        try:
            return ChatOllama(model="llama3.1:8b-instruct-q4_K_M")
        except Exception as e:
            raise RuntimeError(f"Model tidak ditemukan: {e}")
    else:
        raise ValueError("Mode LLM tidak dikenali. Gunakan 'gemini' atau 'ollama'.")


# ===================== FEW SHOT EXAMPLES =====================
few_shot_examples = """
Contoh 1:
Kolom-kolom: article_number, content
Data: [('3', 'Setiap orang dilarang melakukan akses tanpa hak terhadap sistem elektronik milik orang lain.')]

Pertanyaan: Apa isi dari Pasal 3?
Jawaban: Setiap orang dilarang melakukan akses tanpa hak terhadap sistem elektronik milik orang lain.
Referensi: Pasal 3 - Setiap orang dilarang melakukan akses tanpa hak terhadap sistem elektronik milik orang lain.

Contoh 2:
Kolom-kolom: name, definition
Data: [('Tanda Tangan Elektronik', 'Tanda Tangan Elektronik adalah tanda tangan yang terdiri atas Informasi Elektronik yang dilekatkan...')]

Pertanyaan: Apa itu Tanda Tangan Elektronik?
Jawaban: Tanda Tangan Elektronik adalah tanda tangan yang terdiri atas Informasi Elektronik yang dilekatkan pada Informasi Elektronik lainnya sebagai alat verifikasi dan autentikasi.
Referensi: Definisi Tanda Tangan Elektronik
"""

# ===================== PROMPT TEMPLATE =====================
_base_template = """Kamu adalah asisten ahli hukum Indonesia.

{few_shot_section}

Berikut adalah hasil data dari query database:
Kolom-kolom: {columns}
Data: {rows}

Jawablah pertanyaan berikut dengan bahasa Indonesia yang alami, profesional, dan akurat:
{question}

Gunakan petunjuk berikut:
- Jangan mengulang isi data, cukup sebutkan jawaban yang ditanyakan.
- Jangan menambahkan narasi pembuka seperti "berdasarkan data", "terdapat dalam database", atau "dari hasil query".
- Jika dalam data ditemukan referensi tidak eksplisit seperti "peraturan ini", gantilah dengan nama peraturan lengkap dari kolom "title", "number", "year", dan "short_type".
- Jawaban harus berdasarkan data di atas, dan tidak boleh menambahkan informasi umum yang tidak ada di data.
- Jika data terdiri dari beberapa item, sebutkan semuanya sesuai pertanyaan.
- Gunakan bahasa Indonesia formal, profesional, dan tidak bertele-tele.

Format jawaban yang harus kamu berikan:

Jawaban: [Tuliskan jawaban alami yang menjawab pertanyaan secara lengkap]

Referensi: [Tuliskan referensi pasal, isi pasal, atau bagian dari data yang relevan sebagai sumber jawaban]
"""

# Prompt zero-shot (tanpa contoh)
answer_prompt_zero = PromptTemplate(
    input_variables=["columns", "rows", "question"],
    template=_base_template.replace("{few_shot_section}", "")
)

# Prompt few-shot (pakai contoh)
answer_prompt_few = PromptTemplate(
    input_variables=["columns", "rows", "question"],
    template=_base_template.replace("{few_shot_section}", few_shot_examples)
)


def get_sql_chain(llm, mode="zero-shot"):
    prompt = answer_prompt_few if mode == "few-shot" else answer_prompt_zero
    return LLMChain(llm=llm, prompt=prompt)

# Build chain
# answer_chain_zero = LLMChain(llm=llm, prompt=answer_prompt_zero)
# answer_chain_few = LLMChain(llm=llm, prompt=answer_prompt_few)

# ===================== MAIN FUNCTION =====================
def generate_answer(columns, rows, question, mode: str = "zero-shot", llm_mode: str = "gemini"):
    """
    Mengubah hasil query menjadi jawaban bahasa alami berdasarkan pertanyaan awal.
    Pilih LLM via llm_mode: "gemini" atau "ollama"
    Pilih prompt mode: "zero-shot" atau "few-shot"
    """
    rows_text = "\n".join([str(row) for row in rows])
    columns_text = ", ".join(columns)

    llm = init_llm(llm_mode)
    chain = get_sql_chain(llm, mode=mode)

    return chain.run(columns=columns_text, rows=rows_text, question=question).strip()
