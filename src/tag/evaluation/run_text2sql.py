"""Text2SQL evaluation workflow for TAG"""

import time
from typing import List, Tuple
from ragas import EvaluationDataset
from tqdm import tqdm

# Ganti ini dengan import aktual dari tempat kamu menyimpan fungsi generate_sql
from ..src.text2sqlchain import generate_sql  # <- sesuaikan path-nya

def run_text2sql_workflow(
    evaluation_dataset: EvaluationDataset,
    schema_str: str,
    mode: str = "zero-shot",
    top_k: int = 5,
    verbose: bool = True,
) -> Tuple[EvaluationDataset, List[str]]:
    """
    Menjalankan evaluasi text-to-sql dengan fungsi generate_sql.
    
    Args:
        evaluation_dataset (EvaluationDataset): dataset evaluasi berisi pertanyaan
        schema_str (str): deskripsi skema tabel database
        mode (str): 'zero-shot' atau 'few-shot'
        top_k (int): batas jumlah hasil query SQL
        verbose (bool): tampilkan progress bar atau tidak

    Returns:
        Tuple: (dataset yang sama, list hasil SQL yang dihasilkan)
    """
    generated_sql_results = []

    for data in tqdm(
        iterable=evaluation_dataset,
        desc=f"Running text2sql evaluation mode={mode}",
        disable=not verbose,
    ):
        try:
            sql = generate_sql(
                schema=schema_str,
                question=data.user_input,
                top_k=top_k,
                mode=mode,
            )
        except Exception as e:
            sql = f"-- ERROR: {str(e)}"

        generated_sql_results.append(sql)

        # Jika perlu delay untuk hindari rate limit
        # time.sleep(1)

    return evaluation_dataset, generated_sql_results
