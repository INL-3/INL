import re

from dotenv import load_dotenv, find_dotenv
from datasets import Dataset
import ast
import os
load_dotenv(find_dotenv())
from openai import OpenAI

def generate_context_gpt(question, answers):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    Twoim zadaniem jest wygenerować JEDNO zdanie kontekstu faktograficznego.
    WARUNKI:
    1. Musi zawierać DOSŁOWNIE jedną z odpowiedzi: {answers}
    2. Odpowiedź musi wystąpić w NIEZMIENIONEJ formie (substring).
    3. Nie parafrazuj odpowiedzi.
    4. Nie dodawaj prefiksów typu "Odpowiedź brzmi".
    Pytanie: {question}
    Kontekst:
    """
    resp = client.responses.create(
        input=prompt,
        model="gpt-4.1-mini",
        max_output_tokens=200
    )
    return resp.output_text.strip()


def answer_in_context_row(row):
    ctx = str(row["context"]).lower()
    return any(str(a).lower() in ctx for a in row["answers"])


def safe_literal_eval(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return [x]


def to_squad_style(df):
    data = {"question": [], "context": [], "answers": []}
    for _, row in df.iterrows():
        context = row["context"]
        answer_texts = row["answers"]
        texts, starts = [], []
        for ans in answer_texts:
            start = context.find(ans)
            if start != -1:
                texts.append(ans)
                starts.append(start)
        if not texts:
            continue
        data["question"].append(row["question"])
        data["context"].append(context)
        data["answers"].append({"text": texts, "answer_start": starts})
    return Dataset.from_dict(data)


def _normalize_poleval(text):
    text = text.lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def poleval_accuracy_strict(df, qa_pipe):
    correct = 0

    for row in df.itertuples():
        out = qa_pipe(
            question=row.question,
            context=row.context,
            topk=1
        )

        pred = _normalize_poleval(out["answer"])
        golds = [_normalize_poleval(a) for a in row.answers]

        if pred in golds:
            correct += 1

    return correct / len(df)


def poleval_accuracy_context(df, qa_pipe):
    correct = 0
    for row in df.itertuples():
        out = qa_pipe(question=row.question, context=row.context, topk=1)
        if isinstance(out, list):
            out = out[0]

        pred = _normalize_poleval(out["answer"])
        golds = [_normalize_poleval(a) for a in row.answers]

        if any(pred in g or g in pred for g in golds):
            correct += 1

    return correct / len(df)