import os
import sys
import chainlit as cl

# Setup path ke src/
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, "../../src"))
sys.path.append(src_path)

# Import pipeline TAG kamu
# Contoh: dari src/tag/pipeline.py
from tag.pipeline import qa_chain  # Pastikan ada fungsi ini di pipeline-mu

# Chainlit starter
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="public\idea.svg",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="public\idea.svg",
            ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="public\idea.svg",
            ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="",
            )
        ]

async def tag_on_message(query: str):
    query = message.content

    # --- Masukkan Pipeline TAG kamu di sini ---
    try:
        result = qa_chain.invoke(query)  # Ganti dengan pipeline kamu
    except Exception as e:
        result = f"❌ Terjadi kesalahan: {e}"

    await cl.Message(content=result).send()
