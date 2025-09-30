# ==============================================================================
# The Living Brain: A Trilogy of Cognition
# Foundational Cognitive Architecture For A Self-Evolving Digital Mind
#
# Author: Siddhartha Sharma | PALACE-SS
#
# This file serves as the proof-of-concept API for the architecture
# detailed in the accompanying paper.
# ==============================================================================

# --- Imports ---
import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

# --- Load Environment Variables ---
load_dotenv()

# --- Application Setup ---
app = FastAPI(
    title="The Living Brain API",
    description="A proof-of-concept API for a neuro-symbolic cognitive architecture.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Clients & Models ---
# Create expensive objects once to be reused across requests for efficiency.

# LLM for general reasoning and RAG
llm = OllamaLLM(model="llama3:8b")

# LLM specifically for graph extraction tasks
graph_extraction_llm = OllamaLLM(model="llama3:8b")

# Embeddings model for vector store
embeddings = OllamaEmbeddings(model="llama3:8b")

# Neo4j Graph Connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# ChromaDB Vector Store
# This will load the existing database if it exists, or be created upon ingestion.
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# --- Pydantic Models for API Requests ---
class PromptRequest(BaseModel):
    prompt: str

class SynthesizeRequest(BaseModel):
    topic: str

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
def read_root():
    """Root endpoint to welcome users."""
    return {"message": "Welcome to the Living Brain API"}

# ------------------------------------------------------------------------------
# ACT I: The Feynman AI - Answering questions with grounded understanding.
# ------------------------------------------------------------------------------
@app.post("/api/prompt")
async def generate_response(request: PromptRequest):
    """
    This endpoint emulates the Feynman AI. It uses a hybrid RAG approach to answer questions.
    - For factual recall, it uses the Vector Store (System 1).
    - For relational questions, it uses the Knowledge Graph (System 2).
    """
    prompt_text = request.prompt
    
    # --- The Decider ---
    # A simple keyword-based decider to choose the appropriate reasoning engine.
    relationship_keywords = ["relationship", "connect", "link", "between", "how does"]
    
    if any(keyword in prompt_text.lower() for keyword in relationship_keywords):
        # --- Use Knowledge Graph (System 2: Logical Brain) for relational questions ---
        print("--- Using Knowledge Graph to answer ---")
        
        graph_chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            verbose=True
        )
        result = await graph_chain.ainvoke({"query": prompt_text})
        return {"response": result.get("result")}
        
    else:
        # --- Use Vector Store (System 1: Intuitive Brain) for factual recall ---
        print("--- Using Vector Store to answer ---")
        
        retriever = vectorstore.as_retriever()
        
        system_prompt = (
            "You are an intelligent assistant. Use the following context to "
            "answer the user's question. If you don't know the answer, say you "
            "don't know.\n\n{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = await retrieval_chain.ainvoke({"input": prompt_text})
        return {"response": response.get("answer")}

# ------------------------------------------------------------------------------
# Data Ingestion for both Brains
# ------------------------------------------------------------------------------
@app.post("/api/ingest")
async def ingest_document(label: str = Form(...), file: UploadFile = File(...)):
    """
    Ingests a document (PDF or TXT) into both the Intuitive and Logical Brains.
    1. Loads and splits the document.
    2. Ingests chunks into ChromaDB (Vector Store).
    3. Extracts entities and relationships and ingests them into Neo4j (Knowledge Graph).
    """
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    print(f"Ingesting document: {file.filename} with label: {label}")

    # 1. Load the document based on file extension
    loader = PyPDFLoader(file_path) if file.filename.lower().endswith(".pdf") else TextLoader(file_path)
    documents = loader.load()

    # 2. Split the document into chunks for vectorization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 3. Create embeddings and store in ChromaDB (The Intuitive Brain)
    vectorstore.add_documents(documents=splits)
    print(f"Successfully ingested {len(splits)} chunks into ChromaDB.")

    # 4. Extract and Store Triples in Neo4j (The Logical Brain)
    print(f"Extracting entities and relationships for the '{label}' graph...")
    
    extraction_prompt = PromptTemplate.from_template(
        "You are an expert at extracting information. From the following text, "
        "extract entities and relationships as a list of triples. "
        "Format each triple as (Head, RELATION, Tail). "
        "Example: (Marie Curie, FOUND, Radium). "
        "Do not add any explanation or preamble. Just provide the list of triples.\n\n"
        "Text: '''{text}'''"
    )

    triples = []
    for chunk in splits:
        prompt_text = extraction_prompt.format(text=chunk.page_content)
        # Asynchronously call the LLM for each chunk
        chunk_triples_str = await graph_extraction_llm.ainvoke(prompt_text)
        
        # Parse the string response from the current chunk
        for line in chunk_triples_str.strip().split('\n'):
            try:
                # Simple parsing, can be improved for more complex outputs
                head, rel, tail = line.strip().strip('()').split(', ')
                triples.append((head.strip(), rel.strip(), tail.strip()))
            except ValueError:
                continue  # Skip any malformed lines

    # Add extracted triples to the Neo4j graph with the given label
    for head, rel, tail in triples:
        graph.query(
            "MERGE (h:`Entity`:`" + label.capitalize() + "` {name: $head}) "
            "MERGE (t:`Entity`:`" + label.capitalize() + "` {name: $tail}) "
            "MERGE (h)-[:`" + rel.replace(" ", "_").upper() + "`]->(t)",
            params={'head': head, 'tail': tail}
        )
    
    print(f"Successfully added {len(triples)} relationships to the knowledge graph under the label '{label}'.")

    return {"status": "success", "filename": file.filename, "label": label, "chunks": len(splits), "triples": len(triples)}

# ------------------------------------------------------------------------------
# ACT II: The Hegel AI - Synthesizing wisdom from conflict.
# ------------------------------------------------------------------------------
@app.post("/api/synthesize")
async def synthesize_topics(request: SynthesizeRequest):
    """
    This endpoint emulates the Hegel AI. It performs a dialectical reasoning process.
    1. Pulls knowledge from a 'Thesis' and 'Antithesis' graph.
    2. Uses an LLM to identify the core conflict.
    3. Uses a second LLM call to generate a higher-level Synthesis.
    """
    print(f"--- Starting synthesis for topic: {request.topic} ---")

    # 1. Fetch data from both sides of the graph
    thesis_results = graph.query("MATCH (n:Thesis)-[r]->(m:Thesis) RETURN n.name AS head, type(r) AS relation, m.name AS tail LIMIT 25")
    antithesis_results = graph.query("MATCH (n:Antithesis)-[r]->(m:Antithesis) RETURN n.name AS head, type(r) AS relation, m.name AS tail LIMIT 25")
    
    thesis_context = "\n".join([f"({r['head']}, {r['relation']}, {r['tail']})" for r in thesis_results])
    antithesis_context = "\n".join([f"({r['head']}, {r['relation']}, {r['tail']})" for r in antithesis_results])

    if not thesis_context or not antithesis_context:
        return {"error": "Could not find sufficient thesis or antithesis data to perform synthesis."}

    # 2. LLM Call 1: Identify the core conflict
    conflict_prompt = PromptTemplate.from_template(
        "You are a master analyst. Below are two opposing sets of ideas (Thesis and Antithesis) from a knowledge graph.\n\n"
        "Thesis:\n{thesis_context}\n\nAntithesis:\n{antithesis_context}\n\n"
        "Your task is to identify the 2-3 core points of tension between them. Summarize this conflict clearly."
    )
    conflict_chain = conflict_prompt | llm
    print("--- Identifying conflict... ---")
    conflict = await conflict_chain.ainvoke({"thesis_context": thesis_context, "antithesis_context": antithesis_context})

    # 3. LLM Call 2: Generate the Synthesis
    synthesis_prompt = PromptTemplate.from_template(
        "You are a visionary philosopher. You have been presented with two opposing viewpoints and an analysis of their conflict.\n\n"
        "Core Conflict:\n{conflict}\n\n"
        "Your task is to generate a 'Synthesis' - a new, higher-level perspective that resolves or transcends this conflict regarding the topic: '{topic}'."
    )
    synthesis_chain = synthesis_prompt | llm
    print("--- Generating synthesis... ---")
    synthesis = await synthesis_chain.ainvoke({"conflict": conflict, "topic": request.topic})

    return {"thesis_points": len(thesis_results), "antithesis_points": len(antithesis_results), "conflict": conflict, "synthesis": synthesis}

# ------------------------------------------------------------------------------
# ACT III: The Da Vinci AI - Finding creative analogies.
# ------------------------------------------------------------------------------
@app.post("/api/find_analogy")
async def find_analogy(domain_a: str = Form(...), domain_b: str = Form(...)):
    """
    This endpoint emulates the Da Vinci AI. It finds a creative analogy between two domains.
    1. Identifies the most central 'hub' node in each domain within the knowledge graph.
    2. Uses an LLM to generate a creative, analogical leap based on this structural parallel.
    """
    print(f"--- Finding analogy between {domain_a} and {domain_b} ---")

    # 1. Find the central hub node for each domain using node degree
    # This requires the APOC library to be installed in Neo4j.
    hub_a_query = f"MATCH (n:{domain_a.capitalize()}) RETURN n.name AS hub, apoc.node.degree(n) AS degree ORDER BY degree DESC LIMIT 1"
    hub_b_query = f"MATCH (n:{domain_b.capitalize()}) RETURN n.name AS hub, apoc.node.degree(n) AS degree ORDER BY degree DESC LIMIT 1"
    
    hub_a_result = graph.query(hub_a_query)
    hub_b_result = graph.query(hub_b_query)

    hub_a = hub_a_result[0]['hub'] if hub_a_result else None
    hub_b = hub_b_result[0]['hub'] if hub_b_result else None

    if not hub_a or not hub_b:
        return {"error": "Could not find central hub concepts in one or both domains. Ensure data is ingested with these labels and APOC is installed."}

    print(f"Found structural analogs: '{hub_a}' (Domain A) and '{hub_b}' (Domain B)")

    # 2. The "Metaphorical Prompt" for the LLM
    analogy_prompt = PromptTemplate.from_template(
        "You are a creative genius in the style of Leonardo da Vinci. "
        "Your task is to find a deep, insightful analogy between two seemingly unrelated concepts.\n\n"
        "In the domain of {domain_a}, the concept '{hub_a}' is a central structural hub.\n"
        "In the domain of {domain_b}, the concept '{hub_b}' is also a central structural hub.\n\n"
        "Based on this structural parallel, generate a creative and illuminating analogy. "
        "Explore the question: 'What can {domain_b} learn from {domain_a}?' and explain the novel insights derived from this connection."
    )
    
    analogy_chain = analogy_prompt | llm
    print("--- Generating creative analogy... ---")
    analogy = await analogy_chain.ainvoke({
        "domain_a": domain_a,
        "hub_a": hub_a,
        "domain_b": domain_b,
        "hub_b": hub_b
    })

    return {"hub_a": hub_a, "hub_b": hub_b, "analogy": analogy}
