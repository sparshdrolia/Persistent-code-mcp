"""
LlamaIndex Manager Module

Manages the LlamaIndex integration for the persistent-code knowledge graph.
Provides utilities for semantic search, embedding generation, and knowledge graph operations.
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
import json

# LlamaIndex imports
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaIndexManager:
    """Manager for LlamaIndex integration."""
    
    def __init__(self, 
                 project_name: str, 
                 project_dir: str,
                 config_instance,
                 triple_extract_fn: Callable = None):
        """Initialize the LlamaIndex manager.
        
        Args:
            project_name: Name of the project
            project_dir: Directory for project storage
            config_instance: Configuration instance
            triple_extract_fn: Function to extract triples from documents
        """
        self.project_name = project_name
        self.project_dir = project_dir
        self.config = config_instance
        self.triple_extract_fn = triple_extract_fn
        
        # LlamaIndex components
        self.kg_index = None
        self.embed_model = None
        
        # Initialize LlamaIndex
        self._initialize_llama_index()
    
    def _initialize_llama_index(self) -> bool:
        """Initialize LlamaIndex components.
        
        Returns:
            success: Whether initialization was successful
        """
        if not self.config.is_llama_index_enabled():
            logger.info("LlamaIndex integration is disabled in configuration")
            return False
        
        try:
            # Initialize embeddings model using configuration
            model_name = self.config.get_embedding_model()
            logger.info(f"Initializing embedding model: {model_name}")
            self.embed_model = HuggingFaceEmbedding(model_name=model_name)
            
            # Initialize storage context
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
            )
            
            # Initialize the knowledge graph index
            self.kg_index = KnowledgeGraphIndex(
                [],
                storage_context=storage_context,
                embed_model=self.embed_model,
                kg_triple_extract_fn=self.triple_extract_fn,
                include_embeddings=True,
            )
            
            logger.info(f"LlamaIndex Knowledge Graph initialized for project {self.project_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize LlamaIndex: {str(e)}")
            self.kg_index = None
            self.embed_model = None
            return False
    
    def is_available(self) -> bool:
        """Check if LlamaIndex is available and initialized.
        
        Returns:
            is_available: Whether LlamaIndex is available
        """
        return self.kg_index is not None and self.embed_model is not None
    
    def add_document(self, document: Document) -> bool:
        """Add a document to the knowledge graph.
        
        Args:
            document: LlamaIndex document
            
        Returns:
            success: Whether the document was added successfully
        """
        if not self.is_available():
            return False
        
        try:
            # Add to index
            self.kg_index.insert(document)
            logger.info(f"Added document {document.doc_id} to LlamaIndex KG")
            return True
        except Exception as e:
            logger.warning(f"Failed to add document to LlamaIndex: {str(e)}")
            return False
    
    def update_document(self, document: Document) -> bool:
        """Update a document in the knowledge graph.
        
        Args:
            document: LlamaIndex document
            
        Returns:
            success: Whether the document was updated successfully
        """
        if not self.is_available():
            return False
        
        try:
            # Update in index
            self.kg_index.update(document)
            logger.info(f"Updated document {document.doc_id} in LlamaIndex KG")
            return True
        except Exception as e:
            logger.warning(f"Failed to update document in LlamaIndex: {str(e)}")
            return False
    
    def add_triple(self, subject: str, predicate: str, object_text: str) -> bool:
        """Add a knowledge triple to the graph.
        
        Args:
            subject: Subject entity
            predicate: Relationship predicate
            object_text: Object entity
            
        Returns:
            success: Whether the triple was added successfully
        """
        if not self.is_available():
            return False
        
        try:
            # Add triple to knowledge graph
            self.kg_index.upsert_triplet_and_embedding(
                subject, predicate, object_text
            )
            logger.info(f"Added triple: {subject} {predicate} {object_text}")
            return True
        except Exception as e:
            logger.warning(f"Failed to add triple to LlamaIndex: {str(e)}")
            return False
    
    def semantic_search(self, 
                      query: str, 
                      similarity_top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Perform semantic search using LlamaIndex.
        
        Args:
            query: Search query
            similarity_top_k: Number of results to return
            
        Returns:
            results: List of (score, node) tuples
        """
        if not self.is_available():
            logger.warning("Semantic search unavailable - LlamaIndex not initialized")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Search the index
            retriever = self.kg_index.as_retriever(
                similarity_top_k=similarity_top_k
            )
            
            # Get results
            results = retriever.retrieve(query)
            
            # Process results to return (score, node) tuples
            processed_results = []
            for result in results:
                doc_id = result.node.ref_doc_id
                if doc_id:
                    processed_results.append((result.score, {
                        "id": doc_id,
                        "text": result.node.text,
                        "metadata": result.node.metadata
                    }))
            
            logger.info(f"Semantic search for '{query}' found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {str(e)}")
            return []
    
    def save(self) -> bool:
        """Save the LlamaIndex knowledge graph to disk.
        
        Returns:
            success: Whether the save was successful
        """
        if not self.is_available():
            return False
        
        try:
            # Create persist directory
            persist_dir = os.path.join(self.project_dir, "llama_index")
            os.makedirs(persist_dir, exist_ok=True)
            
            # Save index
            self.kg_index.storage_context.persist(persist_dir=persist_dir)
            
            logger.info(f"Saved LlamaIndex to {persist_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save LlamaIndex: {str(e)}")
            return False
    
    def load(self) -> bool:
        """Load the LlamaIndex knowledge graph from disk.
        
        Returns:
            success: Whether the load was successful
        """
        if not self.config.is_llama_index_enabled():
            logger.info("LlamaIndex integration is disabled in configuration")
            return False
        
        # Check if persist directory exists
        persist_dir = os.path.join(self.project_dir, "llama_index")
        if not os.path.exists(persist_dir):
            logger.info(f"No saved LlamaIndex found at {persist_dir}")
            return False
        
        try:
            # Initialize embeddings model using configuration
            if self.embed_model is None:
                model_name = self.config.get_embedding_model()
                logger.info(f"Initializing embedding model: {model_name}")
                self.embed_model = HuggingFaceEmbedding(model_name=model_name)
            
            # Load storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir
            )
            
            # Load index
            self.kg_index = KnowledgeGraphIndex.from_storage(
                storage_context=storage_context,
                embed_model=self.embed_model,
                kg_triple_extract_fn=self.triple_extract_fn,
                include_embeddings=True,
            )
            
            logger.info(f"Loaded LlamaIndex from {persist_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LlamaIndex: {str(e)}")
            self.kg_index = None
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the LlamaIndex integration.
        
        Returns:
            status: Status information
        """
        status = {
            "enabled": self.config.is_llama_index_enabled(),
            "available": self.is_available(),
            "embedding_model": self.config.get_embedding_model() if self.config.is_llama_index_enabled() else None,
        }
        
        # Add index information if available
        if self.is_available() and self.kg_index is not None:
            try:
                docstore = self.kg_index.storage_context.docstore
                doc_count = len(docstore.docs) if hasattr(docstore, 'docs') else 0
                
                status.update({
                    "document_count": doc_count,
                    "index_initialized": True
                })
            except Exception as e:
                status.update({
                    "index_initialized": True,
                    "error": str(e)
                })
        
        return status
