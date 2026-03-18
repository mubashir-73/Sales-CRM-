import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import pandas as pd
from chromadb.config import Settings


class KnowledgeRetriever:
    def __init__(self, kb_path: str = "./knowledge_base"):
        self.kb_path = Path(kb_path)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "chromadb"),
            port=8000
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="revinova_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Content type mapping
        self.content_types = {
            "exploring": ["educational", "overview", "introduction", "guide"],
            "comparing": ["comparison", "features", "specs", "differentiation"],
            "decision_ready": ["pricing", "case_study", "testimonial", "demo"]
        }
    
    def index_knowledge_base(self):
        """Index all knowledge base documents"""
        documents = []
        metadatas = []
        ids = []
        
        # Index CSV files
        for csv_file in self.kb_path.glob("**/*.csv"):
            df = pd.read_csv(csv_file)
            for idx, row in df.iterrows():
                doc_text = " ".join([f"{col}: {val}" for col, val in row.items()])
                doc_id = hashlib.md5(f"{csv_file.name}_{idx}".encode()).hexdigest()
                
                documents.append(doc_text)
                metadatas.append({
                    "source": csv_file.name,
                    "type": self._infer_content_type(csv_file.name),
                    "row": idx,
                    "format": "csv"
                })
                ids.append(doc_id)
        
        # Index text/markdown files
        for txt_file in self.kb_path.glob("**/*.{txt,md}"):
            content = txt_file.read_text()
            doc_id = hashlib.md5(txt_file.name.encode()).hexdigest()
            
            documents.append(content)
            metadatas.append({
                "source": txt_file.name,
                "type": self._infer_content_type(txt_file.name),
                "format": "text"
            })
            ids.append(doc_id)
        
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Indexed {len(documents)} documents")
    
    def _infer_content_type(self, filename: str) -> str:
        """Infer content type from filename"""
        filename_lower = filename.lower()
        
        if any(kw in filename_lower for kw in ["price", "cost", "case", "testimonial"]):
            return "decision_ready"
        elif any(kw in filename_lower for kw in ["compare", "vs", "feature", "spec"]):
            return "comparing"
        else:
            return "exploring"
    
    def retrieve(self, query: str, intent_stage: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents with citation"""
        
        # Filter by content type matching intent stage
        relevant_types = self.content_types.get(intent_stage, ["educational"])
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,  # Get more to filter
            where={"type": {"$in": relevant_types}} if intent_stage != "uncertain" else None
        )
        
        # Format results with citations
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            doc = {
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "relevance_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "citation": {
                    "source": results['metadatas'][0][i]['source'],
                    "type": results['metadatas'][0][i]['type']
                }
            }
            
            # Verify semantic relevance with LLM
            if self._verify_relevance(query, doc['content']):
                retrieved_docs.append(doc)
                
            if len(retrieved_docs) >= top_k:
                break
        
        return retrieved_docs
    
    def _verify_relevance(self, query: str, document: str) -> bool:
        """Verify document relevance using LLM"""
        prompt = f"""Does this document answer or relate to the query?

Query: {query}

Document: {document[:500]}

Answer only 'yes' or 'no'."""

        try:
            import requests
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
                },
                timeout=10
            )
            
            result = response.json()["response"].strip().lower()
            return "yes" in result
            
        except Exception as e:
            print(f"Relevance verification error: {e}")
            return True  # Default to including if verification fails
