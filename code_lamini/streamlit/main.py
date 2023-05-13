import streamlit
import os
from lamini import LLM, Type, Context, get_embedding
import numpy as np
import faiss
from functools import lru_cache

CONTEXT_SIZE = 10
VECTOR_DIMENSION = 128
CHUNK_SIZE=1024

def main():
    llm = build_llm()

    make_app(llm)


def make_app(llm):
    question = streamlit.text_input(
        "Question about Lamini", "How can I run inference on a pythia model?"
    )

    answer = llm.get_answer(question)

    streamlit.write("The answer is", answer.answer)


def build_llm():
    index = build_index()

    pipeline = LLMPipeline(index)

    return pipeline


def build_index():
    faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)

    dataset = load_dataset()

    embeddings = [get_embedding(data) for data in dataset]

    faiss_index.add(np.array(embeddings, dtype=np.float32))

    index = {"faiss_index": faiss_index, "dataset": dataset}

    return index

def load_dataset():
    strings = []
    for root, dirs, files in os.walk("data", topdown=False):
        for filename in files:
            path = os.path.join(root, filename)
            with open(path) as file:
                chunks = chunk(file.read())
                strings.extend(chunks)

    return strings

def chunk(string):
    return string[::CHUNK_SIZE]

class LLMPipeline:
    def __init__(self, index):
        self.llm = LLM(name="code-lamini")
        self.index = index
    
    @lru_cache(maxsize=1024)
    def get_embedding_cached(self, text):
        return get_embedding(text)

    def get_answer(self, question):
        embedding = self.get_embedding_cached(question)
        distances, indices = self.index["faiss_index"].search(np.array([embedding], dtype=np.float32), k=CONTEXT_SIZE)

        similar_data = [self.index["dataset"][index] for index in indices]

        llm_question = LLMQuestion(question=question, similar_data=similar_data)

        return self.llm(input=llm_question, output_type=Answer)


class LLMQuestion(Type):
    question: str = Context("a question about the lamini codebase")
    similar_data: list[str] = Context("code related to the question")


class Answer(Type):
    answer: str = Context("answer to the question")
