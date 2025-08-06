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

response= lafilamod.invoke(token)
print(response)




def rewrite_query(query_text):
    prompt = (
        "You are an AI assistant. Rewrite the following user query into a clear, concise search query suitable for retrieving relevant documents. "
        "User Query: " + query_text + "\nRewritten Query:"
    )
    # Call Llama with this prompt (using your local inference code here)
    rewritten = call_llama_model(prompt)
    return rewritten.strip()
