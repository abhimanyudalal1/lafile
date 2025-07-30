from langchain_ollama import OllamaLLM
import langchain
import numpy
import pandas

#initialize model
lafilamod= OllamaLLM(
    model="Llama3-latest",
    temperature=0,
    num_ctx=500,
    num_predict=2048,
    verbose=True,
)

from sentence_transformers import SentenceTransformer
embeds=input("Me lord! Please enter the path file so i can be of service..:")
token=embeds
lafilamod.invoke(token)