import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
from datasets import Dataset, load_dataset
from langchain.vectorstores import Pinecone
import pandas as pd
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec

# Set up environment variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") or "pinecone_api_key"
os.environ["LANGCHAIN_API_KEY"] = "langchain_api_key"

# Initialize SentenceTransformer model for local embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to initialize Pinecone and create the index
def initialize_pinecone():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = 'llama-3-rag'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

# Function to load dataset and upload to Pinecone
def load_and_upload_data(file_path, index):
    df = pd.read_json(file_path, lines=True)
    dataset = Dataset.from_pandas(df)
    data = dataset.to_pandas()
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()]
        texts = [x['chunk'] for _, x in batch.iterrows()]
        embeds = embed_model.encode(texts, convert_to_tensor=False)
        metadata = [{'text': x['chunk'], 'source': x['source'], 'title': x['title']} for _, x in batch.iterrows()]
        index.upsert(vectors=zip(ids, embeds, metadata))
    return index, dataset

# Function to prepare dataset for fine-tuning
def prepare_dataset_for_finetuning(dataset):
    def generate_prompt(example):
        prompt = f"Title: {example['title']}\nSummary: {example['summary']}\nQuestion: What is the main finding of this paper?\nAnswer:"
        response = f"The main finding of the paper is {example['chunk']}"
        return {"prompt": prompt, "response": response}

    prepared_dataset = dataset.map(generate_prompt)
    prepared_dataset = prepared_dataset.remove_columns([col for col in prepared_dataset.column_names if col not in ['prompt', 'response']])
    return prepared_dataset

# Function to fine-tune the model
def fine_tune_model(dataset):
    model_name = "llama3"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)
    
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    
    training_args = TrainingArguments(
        output_dir="./lora_llama3_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {'input_ids': torch.stack([torch.LongTensor(f['input_ids']) for f in data]),
                                    'attention_mask': torch.stack([torch.LongTensor(f['attention_mask']) for f in data]),
                                    'labels': torch.stack([torch.LongTensor(f['input_ids']) for f in data])}
    )
    
    trainer.train()
    model.save_pretrained("./lora_llama3_finetuned")

# Initialize the local Ollama model
ollama_model = Ollama()

def chatbot_interaction(messages):
    formatted_messages = "\n".join([f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in messages])
    system_prompt = "You are an AI assistant specialized in Llama and related language models research. Provide detailed and accurate information about research papers, their findings, and authors in this field."
    full_prompt = f"{system_prompt}\n\n{formatted_messages}\nHuman: {messages[-1].content}\nAI:"
    try:
        response = ollama_model(full_prompt)
        return AIMessage(content=response)
    except Exception as e:
        return AIMessage(content=f"Error: {str(e)}")

# Streamlit app layout
def main():
    st.title("Llama3 Chatbot")
    
    # Initialize Pinecone index
    index = initialize_pinecone()
    
    # Upload dataset to Pinecone and prepare for fine-tuning
    file_path = "dataset/train.jsonl"
    index, dataset = load_and_upload_data(file_path, index)
    
    # Prepare dataset for fine-tuning
    prepared_dataset = prepare_dataset_for_finetuning(dataset)
    
    # Fine-tune the model (comment out after first run)
    # fine_tune_model(prepared_dataset)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            SystemMessage(content="You are a helpful assistant specialized in Llama and related language models research."),
            HumanMessage(content="Hi AI, can you tell me about recent developments in Llama models?"),
            AIMessage(content="Certainly! I'd be happy to discuss recent developments in Llama models. Llama, developed by Meta AI, has seen significant advancements. Some key points include...")
        ]

    # Display chat history
    for msg in st.session_state["messages"]:
        if isinstance(msg, HumanMessage):
            st.write(f"**Human:** {msg.content}")
        else:
            st.write(f"**AI:** {msg.content}")

    # User input
    user_input = st.text_input("You: ", key="user_input")

    if st.button("Send"):
        if user_input:
            st.session_state["messages"].append(HumanMessage(content=user_input))
            res = chatbot_interaction(st.session_state["messages"])
            st.session_state["messages"].append(res)
            st.experimental_rerun()

if __name__ == "__main__":
    main()