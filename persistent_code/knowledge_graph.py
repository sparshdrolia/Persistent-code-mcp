"""
Knowledge Graph Module

Implements the core knowledge graph functionality for storing code components
and their relationships using LlamaIndex.
"""

import os
import json
import datetime
import networkx as nx
import tempfile
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from enum import Enum
import uuid
import logging

# Import configuration
from .config import config as config_instance

# Import LlamaIndex manager
from .llama_index_manager import LlamaIndexManager

# LlamaIndex imports (for Document type)
from llama_index.core import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentType(str, Enum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    MODULE = "module"

class ComponentStatus(str, Enum):
    PLANNED = "planned"
    PARTIAL = "partial"
    IMPLEMENTED = "implemented"

class RelationshipType(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"

class KnowledgeGraph:
    """A graph-based storage for code components and their relationships using LlamaIndex."""
    
    def __init__(self, project_name: str, storage_dir: str = None):
        """Initialize a new knowledge graph for a project.
        
        Args:
            project_name: Name of the project
            storage_dir: Directory to store the graph data (default: ./storage)
        """
        self.project_name = project_name
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Project specific directory
        self.project_dir = os.path.join(self.storage_dir, project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Graph file paths
        self.nx_graph_file = os.path.join(self.project_dir, f"{project_name}_nx_graph.json")
        self.components_file = os.path.join(self.project_dir, f"{project_name}_components.json")
        
        # Initialize both graph representations
        # 1. NetworkX for structural relationships (inheritance, imports, etc.)
        self.graph = nx.DiGraph()
        
        # 2. LlamaIndex for semantic knowledge
        self.llama_index_manager = LlamaIndexManager(
            project_name=project_name,
            project_dir=self.project_dir,
            config_instance=config_instance,
            triple_extract_fn=self._extract_triples
        )
        
        # Keep backward compatibility with direct kg_index access
        self.kg_index = self.llama_index_manager.kg_index if self.llama_index_manager.is_available() else None
        self.embed_model = self.llama_index_manager.embed_model if self.llama_index_manager.is_available() else None
        
        # Component memory
        self.components = {}
        
        # Load existing graph if available
        self.load()
    
    def _extract_triples(self, document: Document) -> List[Tuple[str, str, str]]:
        """Extract relationship triples for LlamaIndex KG from document metadata.
        
        Args:
            document: LlamaIndex document with component data in metadata
            
        Returns:
            List of (subject, predicate, object) triples
        """
        triples = []
        
        # Extract relationships from metadata
        metadata = document.metadata
        
        # Basic component information
        component_id = metadata.get("id")
        component_name = metadata.get("name")
        component_type = metadata.get("type")
        
        if not (component_id and component_name and component_type):
            return triples
        
        # Add basic type information
        triples.append((component_name, "is_a", component_type))
        
        # Add status information
        status = metadata.get("status")
        if status:
            triples.append((component_name, "has_status", status))
        
        # Add inheritance relationships
        bases = metadata.get("bases", [])
        for base in bases:
            triples.append((component_name, "inherits_from", base))
        
        # Add containment relationships
        contained_in = metadata.get("contained_in")
        if contained_in:
            triples.append((component_name, "contained_in", contained_in))
        
        # Add any explicit relationships
        relationships = metadata.get("relationships", [])
        for rel in relationships:
            source = rel.get("source", component_name)
            relation = rel.get("relation")
            target = rel.get("target")
            
            if relation and target:
                triples.append((source, relation, target))
        
        return triples
    
    def _component_to_document(self, component_data: Dict[str, Any]) -> Document:
        """Convert a component to a LlamaIndex document.
        
        Args:
            component_data: Component data dictionary
            
        Returns:
            LlamaIndex Document
        """
        # Get component text (code + description)
        component_text = component_data.get("code_text", "")
        component_desc = component_data.get("description", "")
        
        # Combine for better semantic search
        full_text = f"{component_desc}\n\n{component_text}"
        
        # Create document
        doc = Document(
            text=full_text,
            metadata=component_data,
            doc_id=component_data.get("id")
        )
        
        return doc
    
    def add_component(self, 
                      name: str, 
                      component_type: ComponentType, 
                      code_text: str = "", 
                      status: ComponentStatus = ComponentStatus.PLANNED, 
                      description: str = "", 
                      metadata: Dict[str, Any] = None) -> str:
        """Add a new component to the graph.
        
        Args:
            name: Name of the component
            component_type: Type of component (file, class, function, etc.)
            code_text: The actual code text (optional)
            status: Implementation status (planned, partial, implemented)
            description: Semantic description of the component
            metadata: Additional metadata as key-value pairs
            
        Returns:
            component_id: Unique identifier for the component
        """
        component_id = str(uuid.uuid4())
        
        # Create component data
        component_data = {
            "id": component_id,
            "name": name,
            "type": component_type,
            "status": status,
            "description": description,
            "code_text": code_text,
            "created_at": datetime.datetime.now().isoformat(),
            "last_modified": datetime.datetime.now().isoformat(),
            "version": 1,
            "metadata": metadata or {}
        }
        
        # Add to NetworkX graph
        self.graph.add_node(component_id, **component_data)
        
        # Store in components dictionary
        self.components[component_id] = component_data
        
        # Add to LlamaIndex knowledge graph if available
        if self.llama_index_manager.is_available():
            # Convert to document
            doc = self._component_to_document(component_data)
            
            # Add to index
            self.llama_index_manager.add_document(doc)
        
        # Save the updated graph
        self.save()
        
        return component_id
    
    def update_component(self, 
                        component_id: str, 
                        code_text: Optional[str] = None, 
                        status: Optional[ComponentStatus] = None,
                        description: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing component in the graph.
        
        Args:
            component_id: ID of the component to update
            code_text: New code text (if changed)
            status: New implementation status (if changed)
            description: New description (if changed)
            metadata: New or updated metadata
            
        Returns:
            success: Whether the update was successful
        """
        if component_id not in self.graph:
            return False
        
        # Get current data
        component_data = dict(self.graph.nodes[component_id])
        
        # Update fields if provided
        if code_text is not None:
            component_data["code_text"] = code_text
        
        if status is not None:
            component_data["status"] = status
            
        if description is not None:
            component_data["description"] = description
            
        if metadata is not None:
            component_data["metadata"].update(metadata)
        
        # Update modification time and version
        component_data["last_modified"] = datetime.datetime.now().isoformat()
        component_data["version"] += 1
        
        # Update in NetworkX graph
        self.graph.nodes[component_id].update(component_data)
        
        # Update in components dictionary
        self.components[component_id] = component_data
        
        # Update in LlamaIndex if available
        if self.llama_index_manager.is_available():
            # Convert to document
            doc = self._component_to_document(component_data)
            
            # Update in index
            self.llama_index_manager.update_document(doc)
        
        # Save the updated graph
        self.save()
        
        return True
    
    def add_relationship(self,
                       source_id: str,
                       target_id: str,
                       relationship_type: RelationshipType,
                       metadata: Dict[str, Any] = None) -> str:
        """Create a relationship between two components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            relationship_type: Type of relationship
            metadata: Additional metadata for the relationship
            
        Returns:
            relationship_id: Unique identifier for the relationship
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None
        
        relationship_id = str(uuid.uuid4())
        
        # Create edge data
        edge_data = {
            "id": relationship_id,
            "type": relationship_type,
            "created_at": datetime.datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add edge to NetworkX graph
        self.graph.add_edge(source_id, target_id, **edge_data)
        
        # Add relationship to LlamaIndex if available
        if self.llama_index_manager.is_available():
            # Get component names
            source_name = self.graph.nodes[source_id].get("name")
            target_name = self.graph.nodes[target_id].get("name")
            relation = str(relationship_type).lower()
            
            # Add triple to knowledge graph
            self.llama_index_manager.add_triple(source_name, relation, target_name)
        
        # Save the updated graph
        self.save()
        
        return relationship_id
    
    def get_component(self, component_id: str, include_related: bool = False) -> Dict[str, Any]:
        """Retrieve a component by ID.
        
        Args:
            component_id: ID of the component
            include_related: Whether to include related components
            
        Returns:
            component: Component data including code and metadata
        """
        if component_id not in self.graph:
            return None
        
        # Get component data from NetworkX
        component = dict(self.graph.nodes[component_id])
        
        if include_related:
            # Get incoming relationships
            incoming = []
            for source_id, _, edge_data in self.graph.in_edges(component_id, data=True):
                source_data = dict(self.graph.nodes[source_id])
                incoming.append({
                    "component": source_data,
                    "relationship": edge_data
                })
            component["incoming_relationships"] = incoming
            
            # Get outgoing relationships
            outgoing = []
            for _, target_id, edge_data in self.graph.out_edges(component_id, data=True):
                target_data = dict(self.graph.nodes[target_id])
                outgoing.append({
                    "component": target_data,
                    "relationship": edge_data
                })
            component["outgoing_relationships"] = outgoing
        
        return component
    
    def find_related_components(self, 
                              component_id: str, 
                              relationship_types: Optional[List[RelationshipType]] = None,
                              depth: int = 1) -> List[Dict[str, Any]]:
        """Find components related to a given component.
        
        Args:
            component_id: ID of the component
            relationship_types: Types of relationships to consider (None for all)
            depth: How many levels of relationships to traverse
            
        Returns:
            related_components: List of related components
        """
        if component_id not in self.graph:
            return []
        
        # Helper function to check if relationship matches the types
        def matches_type(edge_data):
            if relationship_types is None:
                return True
            return edge_data.get("type") in relationship_types
        
        # Use NetworkX to get subgraph within depth
        if depth == 1:
            # Direct connections only
            related_ids = set()
            
            # Check outgoing edges
            for _, target_id, edge_data in self.graph.out_edges(component_id, data=True):
                if matches_type(edge_data):
                    related_ids.add(target_id)
            
            # Check incoming edges
            for source_id, _, edge_data in self.graph.in_edges(component_id, data=True):
                if matches_type(edge_data):
                    related_ids.add(source_id)
        else:
            # Use BFS or DFS to find components within depth
            related_ids = set()
            
            # Helper function for DFS
            def dfs(node_id, current_depth):
                if current_depth > depth:
                    return
                
                # Check outgoing edges
                for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
                    if matches_type(edge_data) and target_id != component_id:
                        related_ids.add(target_id)
                        dfs(target_id, current_depth + 1)
                
                # Check incoming edges
                for source_id, _, edge_data in self.graph.in_edges(node_id, data=True):
                    if matches_type(edge_data) and source_id != component_id:
                        related_ids.add(source_id)
                        dfs(source_id, current_depth + 1)
            
            # Start DFS from the component
            dfs(component_id, 1)
        
        # Get component data for all related IDs
        related_components = []
        for related_id in related_ids:
            related_components.append(dict(self.graph.nodes[related_id]))
        
        return related_components
    
    def search_code(self, 
                  query: str, 
                  component_types: Optional[List[ComponentType]] = None,
                  limit: int = 10) -> List[Dict[str, Any]]:
        """Search the codebase semantically.
        
        Args:
            query: Search query
            component_types: Types of components to search
            limit: Maximum number of results
            
        Returns:
            matching_components: Ranked list of matching components
        """
        matches = []
        
        # Apply configured limit if different from default
        config_limit = config_instance.get_similarity_top_k()
        if config_limit != 5:  # Default value in config
            limit = config_limit
        
        # Try semantic search with LlamaIndex if available
        if self.llama_index_manager.is_available():
            # Use LlamaIndex for semantic search
            logger.info(f"Performing semantic search for: {query}")
            
            # Get results from LlamaIndex manager
            search_results = self.llama_index_manager.semantic_search(
                query=query,
                similarity_top_k=limit * 2  # Get more than needed, then filter
            )
            
            # Process results
            for score, result in search_results:
                doc_id = result.get('id')
                if doc_id and doc_id in self.components:
                    component_data = self.components[doc_id]
                    
                    # Filter by component type if specified
                    if component_types and component_data.get("type") not in component_types:
                        continue
                    
                    # Add score
                    matches.append((score, component_data))
            
            # If we got results, return them
            if matches:
                # Sort by score and limit results
                matches.sort(key=lambda x: x[0], reverse=True)
                semantic_results = [m[1] for m in matches[:limit]]
                logger.info(f"Semantic search found {len(semantic_results)} matches")
                return semantic_results
        
        # Fall back to basic text search
        logger.info("Using basic text search")
        for component_id, component_data in self.graph.nodes(data=True):
            # Filter by component type if specified
            if component_types and component_data.get("type") not in component_types:
                continue
            
            # Simple text search in name, description, and code
            score = 0
            query_lower = query.lower()
            
            # Check name
            component_name = component_data.get("name", "").lower()
            if query_lower in component_name:
                name_score = 3
                # Exact match gets higher score
                if query_lower == component_name:
                    name_score = 5
                score += name_score
            
            # Check description
            if query_lower in component_data.get("description", "").lower():
                score += 2
            
            # Check code
            if query_lower in component_data.get("code_text", "").lower():
                score += 1
            
            if score > 0:
                matches.append((score, component_data))
        
        # Sort by score and limit results
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]
    
    def update_status(self, 
                    component_id: str, 
                    new_status: ComponentStatus,
                    notes: str = "") -> bool:
        """Update implementation status of a component.
        
        Args:
            component_id: ID of the component
            new_status: New implementation status
            notes: Optional notes about the status change
            
        Returns:
            success: Whether the update was successful
        """
        if component_id not in self.graph:
            return False
        
        # Update the status
        self.graph.nodes[component_id]["status"] = new_status
        
        # Add status change to history if it doesn't exist
        if "status_history" not in self.graph.nodes[component_id]:
            self.graph.nodes[component_id]["status_history"] = []
        
        # Add status change to history
        self.graph.nodes[component_id]["status_history"].append({
            "status": new_status,
            "timestamp": datetime.datetime.now().isoformat(),
            "notes": notes
        })
        
        # Update in components dictionary
        self.components[component_id] = dict(self.graph.nodes[component_id])
        
        # Update status in LlamaIndex if available
        if self.llama_index_manager.is_available():
            component_name = self.graph.nodes[component_id].get("name")
            status_value = str(new_status).lower()
            
            # Update status triple
            self.llama_index_manager.add_triple(component_name, "has_status", status_value)
        
        # Save the updated graph
        self.save()
        
        return True
    
    def get_project_status(self, 
                         filters: Dict[str, Any] = None,
                         grouping: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve implementation status across the project.
        
        Args:
            filters: Filters to apply to components
            grouping: How to group the results (e.g., by type)
            
        Returns:
            status_summary: Summary of project implementation status
        """
        # Count components by status
        status_counts = {
            ComponentStatus.PLANNED: 0,
            ComponentStatus.PARTIAL: 0,
            ComponentStatus.IMPLEMENTED: 0
        }
        
        # Filtered components
        filtered_components = []
        
        for _, component_data in self.graph.nodes(data=True):
            # Apply filters if specified
            if filters:
                skip = False
                for key, value in filters.items():
                    if key in component_data and component_data[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Count by status
            status = component_data.get("status", ComponentStatus.PLANNED)
            status_counts[status] += 1
            
            # Add to filtered list
            filtered_components.append(component_data)
        
        # Prepare summary
        summary = {
            "total_components": len(filtered_components),
            "status_counts": status_counts,
            "implementation_percentage": 0
        }
        
        # Calculate implementation percentage
        if summary["total_components"] > 0:
            implemented = status_counts[ComponentStatus.IMPLEMENTED]
            partial = status_counts[ComponentStatus.PARTIAL] * 0.5  # Count partial as half implemented
            summary["implementation_percentage"] = (implemented + partial) / summary["total_components"] * 100
        
        # Group components if specified
        if grouping:
            grouped = {}
            for component in filtered_components:
                group_key = component.get(grouping, "unknown")
                if group_key not in grouped:
                    grouped[group_key] = []
                grouped[group_key].append(component)
            
            summary["grouped_components"] = grouped
        else:
            summary["components"] = filtered_components
        
        return summary
    
    def find_next_tasks(self, 
                      priority_type: str = "dependencies",
                      limit: int = 5) -> List[Dict[str, Any]]:
        """Suggest logical next components to implement.
        
        Args:
            priority_type: How to prioritize tasks ("dependencies" or "complexity")
            limit: Maximum number of tasks to suggest
            
        Returns:
            suggested_tasks: List of suggested next tasks
        """
        # Find all planned or partial components
        pending = []
        for component_id, component_data in self.graph.nodes(data=True):
            status = component_data.get("status")
            if status in [ComponentStatus.PLANNED, ComponentStatus.PARTIAL]:
                pending.append((component_id, component_data))
        
        # Sort by priority type
        if priority_type == "dependencies":
            # Prioritize components with all dependencies implemented
            sorted_pending = []
            for component_id, component_data in pending:
                # Check outgoing "depends_on" relationships
                dependencies = []
                for _, target_id, edge_data in self.graph.out_edges(component_id, data=True):
                    if edge_data.get("type") == RelationshipType.DEPENDS_ON:
                        target_status = self.graph.nodes[target_id].get("status")
                        dependencies.append((target_id, target_status))
                
                # Calculate a dependency score (higher means more deps are implemented)
                total_deps = len(dependencies)
                implemented_deps = sum(1 for _, status in dependencies 
                                      if status == ComponentStatus.IMPLEMENTED)
                
                if total_deps == 0:
                    dep_score = 1.0  # No dependencies, high priority
                else:
                    dep_score = implemented_deps / total_deps
                
                # Add to sorted list
                sorted_pending.append((dep_score, component_data))
            
            # Sort by dependency score (highest first)
            sorted_pending.sort(key=lambda x: x[0], reverse=True)
            suggested = [comp for _, comp in sorted_pending[:limit]]
            
        elif priority_type == "complexity":
            # Prioritize less complex components first
            sorted_pending = []
            for _, component_data in pending:
                # Use code length as a simple complexity measure
                code_text = component_data.get("code_text", "")
                complexity = component_data.get("metadata", {}).get("complexity", len(code_text))
                
                # Add to sorted list
                sorted_pending.append((complexity, component_data))
            
            # Sort by complexity (lowest first)
            sorted_pending.sort(key=lambda x: x[0])
            suggested = [comp for _, comp in sorted_pending[:limit]]
            
        else:
            # Default to random order
            import random
            suggested = [comp for _, comp in random.sample(pending, min(limit, len(pending)))]
        
        return suggested
    
    def save(self):
        """Save the knowledge graph to disk."""
        # 1. Save NetworkX graph
        self._save_nx_graph()
        
        # 2. Save components dictionary
        self._save_components()
        
        # 3. Save LlamaIndex if available
        if self.llama_index_manager.is_available():
            self.llama_index_manager.save()
    
    def _save_nx_graph(self):
        """Save the NetworkX graph to disk."""
        # Convert NetworkX graph to JSON-serializable format
        data = {
            "project_name": self.project_name,
            "saved_at": datetime.datetime.now().isoformat(),
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            # Convert to serializable format
            serializable_data = {}
            for key, value in node_data.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    serializable_data[key] = value
                else:
                    # Convert non-serializable types to string
                    serializable_data[key] = str(value)
            
            data["nodes"].append(serializable_data)
        
        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            # Convert to serializable format
            serializable_data = {}
            for key, value in edge_data.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    serializable_data[key] = value
                else:
                    # Convert non-serializable types to string
                    serializable_data[key] = str(value)
            
            data["edges"].append({
                "source": source,
                "target": target,
                "data": serializable_data
            })
        
        # Write to file
        with open(self.nx_graph_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved NetworkX graph to {self.nx_graph_file}")
    
    def _save_components(self):
        """Save the components dictionary to disk."""
        # Write to file
        with open(self.components_file, 'w') as f:
            json.dump(self.components, f, indent=2)
        
        logger.info(f"Saved components to {self.components_file}")
    
    def _save_llama_index(self):
        """Save the LlamaIndex knowledge graph to disk."""
        # Create persist directory
        persist_dir = os.path.join(self.project_dir, "llama_index")
        os.makedirs(persist_dir, exist_ok=True)
        
        # Save index
        self.kg_index.storage_context.persist(persist_dir=persist_dir)
        
        logger.info(f"Saved LlamaIndex to {persist_dir}")
    
    def load(self):
        """Load the knowledge graph from disk."""
        # Load NetworkX graph if it exists
        if os.path.exists(self.nx_graph_file):
            self._load_nx_graph()
        
        # Load components if they exist
        if os.path.exists(self.components_file):
            self._load_components()
        
        # Load LlamaIndex if possible
        llama_index_loaded = self.llama_index_manager.load()
        
        # Update references for backward compatibility
        if llama_index_loaded:
            self.kg_index = self.llama_index_manager.kg_index
            self.embed_model = self.llama_index_manager.embed_model
    
    def _load_nx_graph(self):
        """Load the NetworkX graph from disk."""
        # Read from file
        with open(self.nx_graph_file, 'r') as f:
            data = json.load(f)
        
        # Create a new graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge in data.get("edges", []):
            source = edge["source"]
            target = edge["target"]
            edge_data = edge["data"]
            self.graph.add_edge(source, target, **edge_data)
        
        logger.info(f"Loaded NetworkX graph from {self.nx_graph_file}")
    
    def _load_components(self):
        """Load the components dictionary from disk."""
        # Read from file
        with open(self.components_file, 'r') as f:
            self.components = json.load(f)
        
        logger.info(f"Loaded components from {self.components_file}")
    
    def _load_llama_index(self, persist_dir):
        """Load the LlamaIndex knowledge graph from disk.
        
        Args:
            persist_dir: Directory where index is persisted
        """
        # Load storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        
        # Load index
        self.kg_index = KnowledgeGraphIndex.from_storage(
            storage_context=storage_context,
            embed_model=self.embed_model,
            kg_triple_extract_fn=self._extract_triples,
            include_embeddings=True,
        )
        
        logger.info(f"Loaded LlamaIndex from {persist_dir}")
