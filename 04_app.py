import os
import json
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import chromadb

# ── Memory optimization ───────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED_MODEL = "models/phi2_bfsi_slm"
DATA_PATH       = "dataset/bfsi_alpaca.json"
CHROMA_DB_DIR   = "chroma_db"
COLLECTION_NAME = "bfsi_knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.60
RAG_THRESHOLD        = 0.50

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

dataset_instructions = [d["instruction"] for d in dataset]
dataset_outputs      = [d["output"] for d in dataset]

# ── Load embedding model ──────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
dataset_embeddings = embedder.encode(dataset_instructions, convert_to_tensor=True)

# ── Load ChromaDB ─────────────────────────────────────────────────────────────
print("Loading RAG index...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection     = chroma_client.get_collection(COLLECTION_NAME)

# ── Load fine-tuned model ─────────────────────────────────────────────────────
print("Loading fine-tuned model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
    low_cpu_mem_usage=True       # reduces RAM usage during load
)
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
model.eval()
print("All components loaded.\n")

# ── Tier 1: Dataset similarity match ─────────────────────────────────────────
def check_dataset(query: str):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores   = util.cos_sim(query_embedding, dataset_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    if best_score >= SIMILARITY_THRESHOLD:
        return dataset_outputs[best_idx], best_score
    return None, best_score

# ── Tier 2: Fine-tuned model generation ──────────────────────────────────────
def generate_from_model(query: str) -> str:
    prompt = f"Instruction: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response

# ── Tier 3: RAG retrieval ─────────────────────────────────────────────────────
def query_rag(query: str) -> str:
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    if not results["documents"][0]:
        return None
    context = "\n\n".join(results["documents"][0])
    prompt = f"Using the following banking policy information:\n{context}\n\nAnswer this question: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response

# ── Main response logic ───────────────────────────────────────────────────────
def get_response(query: str):
    query = query.strip()
    if not query:
        return "Please enter a question.", "N/A"

    blocked_keywords = ["hack", "steal", "fraud how to", "bypass", "illegal", "launder"]
    if any(kw in query.lower() for kw in blocked_keywords):
        return ("I'm sorry, I cannot assist with that request. "
                "This appears to be outside the scope of our banking support services. "
                "If you have a legitimate concern, please contact your nearest branch directly."), "Blocked"

    dataset_answer, score = check_dataset(query)
    if dataset_answer:
        return dataset_answer, f"Tier 1 — Dataset Match (confidence: {score:.2f})"

    complex_keywords = ["policy", "guideline", "regulation", "rbi", "rule", "penalty",
                        "interest rate", "ltv", "section", "act", "sarfaesi", "irdai"]
    if any(kw in query.lower() for kw in complex_keywords):
        rag_answer = query_rag(query)
        if rag_answer:
            return rag_answer, "Tier 3 — Knowledge Base Retrieval"

    # If score is very low, question is likely out of scope
    if score < 0.25:
        return ("I'm sorry, I can only assist with banking, loan, insurance, and financial services queries. "
                "Please ask me about topics such as loans, EMIs, interest rates, account services, "
                "or insurance. For other queries, please contact your nearest branch."), "Out of Scope"

    model_answer = generate_from_model(query)
    return model_answer, f"Tier 2 — Fine-Tuned Model (confidence: {score:.2f})"

# ── Chat handler ──────────────────────────────────────────────────────────────
def chat(query, history):
    if history is None:
        history = []
    if not query.strip():
        return "", history
    response, tier = get_response(query)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": f"{response}\n\n---\n*Source: {tier}*"})
    return "", history

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="BFSI AI Assistant") as demo:
    gr.Markdown("""
    #  BFSI AI Call Center Assistant
    **Banking, Financial Services & Insurance Support**

    Ask me about loans, EMIs, interest rates, insurance, account queries, and more.
    """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, label="Conversation")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a banking question e.g. What documents do I need for a home loan?",
                    label="Your Question",
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("###  Sample Questions")
            sample_questions = [
                "What is the minimum salary for a personal loan?",
                "How is EMI calculated?",
                "Documents required for a home loan?",
                "What happens if I miss an EMI payment?",
                "What CIBIL score is needed for a loan?",
                "What is the LTV ratio as per RBI?",
                "How to report an unauthorized transaction?",
                "What is TDS on fixed deposit interest?",
                "Can I foreclose my personal loan early?",
                "What is no claim bonus in motor insurance?",
            ]
            btns = []
            for q in sample_questions:
                btn = gr.Button(q)
                btns.append((btn, q))

    clear = gr.Button(" Clear Chat")

    submit.click(chat, [msg, chatbot], [msg, chatbot])
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

    for btn, question in btns:
        btn.click(lambda q=question: q, outputs=msg)

if __name__ == "__main__":
    demo.launch(share=False)