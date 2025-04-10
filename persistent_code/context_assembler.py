"""
Context Assembly System

Responsible for assembling minimal necessary context for coding tasks,
summarizing peripheral components, and prioritizing based on relevance.
"""

from typing import Dict, List, Optional, Any, Set
import logging
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .knowledge_graph import (
    KnowledgeGraph,
    ComponentType,
    ComponentStatus,
    RelationshipType
)

class ContextAssembler:
    """Assembles context for coding tasks from the knowledge graph."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """Initialize the context assembler.
        
        Args:
            knowledge_graph: The knowledge graph to use
        """
        self.knowledge_graph = knowledge_graph
    
    def prepare_context(self,
                      task_description: str,
                      relevant_components: Optional[List[str]] = None,
                      max_tokens: int = 4000) -> Dict[str, Any]:
        """Assemble minimal context for a specific task.
        
        This method uses semantic search in the knowledge graph to find
        components relevant to the task, then assembles them into a
        context object with primary components, related components, and
        summaries if needed to stay within token limits.
        
        Args:
            task_description: Description of the task
            relevant_components: List of component IDs known to be relevant
            max_tokens: Maximum tokens for the context
            
        Returns:
            context: Assembled context information
        """
        # Start with an empty context
        context = {
            "task_description": task_description,
            "primary_components": [],
            "related_components": [],
            "summaries": {}
        }
        
        # If no relevant components specified, find them based on task description
        if not relevant_components:
            # Search for components related to the task
            logger.info(f"Searching for components relevant to: {task_description}")
            search_results = self.knowledge_graph.search_code(
                query=task_description,
                limit=5
            )
            relevant_components = [r["id"] for r in search_results]
            logger.info(f"Found {len(relevant_components)} relevant components")
        
        # Process each relevant component
        token_count = len(task_description.split())
        for component_id in relevant_components:
            # Get the component
            component = self.knowledge_graph.get_component(component_id)
            if not component:
                continue
            
            # Add to primary components
            context["primary_components"].append(component)
            
            # Add tokens for this component
            component_tokens = len(component.get("code_text", "").split())
            token_count += component_tokens
            
            # If we're approaching the token limit, stop adding primary components
            if token_count > max_tokens * 0.7:
                break
        
        # Find related components (dependencies, callers, etc.)
        related_ids = set()
        for component in context["primary_components"]:
            component_id = component["id"]
            
            # Find direct relationships
            related = self.knowledge_graph.find_related_components(
                component_id=component_id,
                depth=1
            )
            
            for related_component in related:
                related_id = related_component["id"]
                if related_id not in related_ids and related_id not in [c["id"] for c in context["primary_components"]]:
                    related_ids.add(related_id)
        
        logger.info(f"Found {len(related_ids)} related components")
        
        # Add related components until we hit the token limit
        for related_id in related_ids:
            # Get the component
            component = self.knowledge_graph.get_component(related_id)
            if not component:
                continue
            
            # Add tokens for this component
            component_tokens = len(component.get("code_text", "").split())
            
            # If adding this would exceed the token limit, summarize instead
            if token_count + component_tokens > max_tokens:
                # Create a summary
                summary = self._summarize_component(component)
                context["summaries"][component["id"]] = summary
                
                # Add tokens for the summary
                token_count += len(summary.split())
            else:
                # Add to related components
                context["related_components"].append(component)
                token_count += component_tokens
            
            # If we've hit the token limit, stop
            if token_count >= max_tokens:
                break
        
        logger.info(f"Assembled context with {len(context['primary_components'])} primary and {len(context['related_components'])} related components ({token_count} tokens)")
        return context
    
    def continue_implementation(self, component_id: str, max_tokens: int = 4000) -> Dict[str, Any]:
        """Provide context to continue implementing a component.
        
        Args:
            component_id: ID of the component to continue implementing
            max_tokens: Maximum tokens for the context
            
        Returns:
            context: Implementation context
        """
        # Get the component
        component = self.knowledge_graph.get_component(component_id)
        if not component:
            return {"error": "Component not found"}
        
        logger.info(f"Preparing context to continue implementing: {component.get('name')}")
        
        # Start with the component itself
        context = {
            "component": component,
            "dependencies": [],
            "dependents": [],
            "related_implementations": [],
            "summaries": {}
        }
        
        # Get dependencies (what this component needs)
        if "outgoing_relationships" in component:
            for rel in component.get("outgoing_relationships", []):
                if rel["relationship"].get("type") in [RelationshipType.DEPENDS_ON, RelationshipType.CALLS]:
                    context["dependencies"].append(rel["component"])
        else:
            # Find dependencies through graph search
            dependencies = self.knowledge_graph.find_related_components(
                component_id=component_id,
                relationship_types=[RelationshipType.DEPENDS_ON, RelationshipType.CALLS],
                depth=1
            )
            context["dependencies"] = dependencies
        
        # Get dependents (what needs this component)
        if "incoming_relationships" in component:
            for rel in component.get("incoming_relationships", []):
                if rel["relationship"].get("type") in [RelationshipType.DEPENDS_ON, RelationshipType.CALLS]:
                    context["dependents"].append(rel["component"])
        else:
            # Find dependents through graph search
            dependents = []
            for node_id in self.knowledge_graph.graph.nodes():
                for _, target_id, edge_data in self.knowledge_graph.graph.out_edges(node_id, data=True):
                    if (target_id == component_id and 
                        edge_data.get("type") in [RelationshipType.DEPENDS_ON, RelationshipType.CALLS]):
                        dependent = self.knowledge_graph.get_component(node_id)
                        if dependent:
                            dependents.append(dependent)
            context["dependents"] = dependents
        
        # Find similar implemented components as examples
        component_type = component.get("type")
        if component_type:
            implemented = []
            for node_id, node_data in self.knowledge_graph.graph.nodes(data=True):
                if (node_data.get("type") == component_type and 
                    node_data.get("status") == ComponentStatus.IMPLEMENTED and
                    node_id != component_id):
                    implemented.append(node_data)
            
            # Sort by similarity (simple name-based similarity for now)
            component_name = component.get("name", "")
            scored = []
            for impl in implemented:
                name_similarity = self._simple_similarity(component_name, impl.get("name", ""))
                scored.append((name_similarity, impl))
            
            # Take top 3 most similar
            scored.sort(key=lambda x: x[0], reverse=True)
            context["related_implementations"] = [s[1] for s in scored[:3]]
        
        # Calculate total tokens
        token_count = 0
        token_count += len(str(component).split())
        
        for dep in context["dependencies"]:
            token_count += len(str(dep).split())
        
        for dep in context["dependents"]:
            token_count += len(str(dep).split())
        
        for impl in context["related_implementations"]:
            token_count += len(str(impl).split())
        
        # If we're over the token limit, summarize some components
        if token_count > max_tokens:
            logger.info(f"Token count ({token_count}) exceeds limit ({max_tokens}), summarizing components")
            # Summarize dependents first
            for i, dep in enumerate(context["dependents"]):
                summary = self._summarize_component(dep)
                context["summaries"][dep["id"]] = summary
                token_count -= len(str(dep).split())
                token_count += len(summary.split())
                context["dependents"][i] = {"id": dep["id"], "name": dep["name"], "summarized": True}
                
                if token_count <= max_tokens:
                    break
            
            # If still over limit, summarize related implementations
            if token_count > max_tokens:
                for i, impl in enumerate(context["related_implementations"]):
                    summary = self._summarize_component(impl)
                    context["summaries"][impl["id"]] = summary
                    token_count -= len(str(impl).split())
                    token_count += len(summary.split())
                    context["related_implementations"][i] = {"id": impl["id"], "name": impl["name"], "summarized": True}
                    
                    if token_count <= max_tokens:
                        break
            
            # If still over limit, summarize dependencies
            if token_count > max_tokens:
                for i, dep in enumerate(context["dependencies"]):
                    summary = self._summarize_component(dep)
                    context["summaries"][dep["id"]] = summary
                    token_count -= len(str(dep).split())
                    token_count += len(summary.split())
                    context["dependencies"][i] = {"id": dep["id"], "name": dep["name"], "summarized": True}
                    
                    if token_count <= max_tokens:
                        break
        
        logger.info(f"Assembled implementation context for {component.get('name')} ({token_count} tokens)")
        return context
    
    def get_implementation_plan(self,
                             component_ids: List[str] = None,
                             dependencies_depth: int = 1) -> Dict[str, Any]:
        """Generate a plan for implementing pending components.
        
        Args:
            component_ids: List of component IDs to include (None for all pending)
            dependencies_depth: How deep to analyze dependencies
            
        Returns:
            plan: Ordered implementation plan
        """
        # Start with an empty plan
        plan = {
            "ordered_components": [],
            "dependency_groups": {},
            "implementation_steps": []
        }
        
        # Get all pending components if none specified
        pending_components = []
        if component_ids:
            for component_id in component_ids:
                component = self.knowledge_graph.get_component(component_id)
                if component and component.get("status") != ComponentStatus.IMPLEMENTED:
                    pending_components.append(component)
        else:
            # Get all planned or partial components
            for node_id, node_data in self.knowledge_graph.graph.nodes(data=True):
                status = node_data.get("status")
                if status in [ComponentStatus.PLANNED, ComponentStatus.PARTIAL]:
                    pending_components.append(node_data)
        
        # No pending components
        if not pending_components:
            return {"error": "No pending components found"}
        
        logger.info(f"Planning implementation for {len(pending_components)} pending components")
        
        # Build a dependency graph
        dep_graph = nx.DiGraph()
        
        # Add all pending components to the graph
        for component in pending_components:
            dep_graph.add_node(component["id"], **component)
        
        # Add dependencies between components
        for component in pending_components:
            component_id = component["id"]
            
            # Get dependencies
            dependencies = []
            for _, target_id, edge_data in self.knowledge_graph.graph.out_edges(component_id, data=True):
                if edge_data.get("type") in [RelationshipType.DEPENDS_ON, RelationshipType.CALLS]:
                    dependencies.append(target_id)
            
            # Add edges for dependencies
            for dep_id in dependencies:
                if dep_id in dep_graph:
                    dep_graph.add_edge(component_id, dep_id)
        
        # Check for cycles
        try:
            # This will raise an exception if there's a cycle
            cycles = list(nx.simple_cycles(dep_graph))
            if cycles:
                plan["warnings"] = [f"Dependency cycle detected: {' -> '.join([dep_graph.nodes[n]['name'] for n in cycle])}" for cycle in cycles]
                logger.warning(f"Detected {len(cycles)} dependency cycles")
        except nx.NetworkXNoCycle:
            # No cycles
            pass
        
        # Get a topological sort if possible (this gives an order where all dependencies come before dependents)
        try:
            ordered_components = list(nx.topological_sort(dep_graph))
            ordered_components.reverse()  # Reverse so dependencies come first
            
            # Add to plan
            for component_id in ordered_components:
                component = self.knowledge_graph.get_component(component_id)
                if component:
                    plan["ordered_components"].append(component)
        except nx.NetworkXUnfeasible:
            # Can't topologically sort due to cycles
            # Just add components in original order
            for component in pending_components:
                plan["ordered_components"].append(component)
        
        # Group components by dependency relationships
        # Components that can be implemented in parallel go in the same group
        dependency_groups = []
        remaining = set(component["id"] for component in pending_components)
        
        while remaining:
            # Find components with no pending dependencies
            group = []
            for component_id in list(remaining):
                has_pending_deps = False
                for _, target_id in dep_graph.out_edges(component_id):
                    if target_id in remaining:
                        has_pending_deps = True
                        break
                
                if not has_pending_deps:
                    group.append(component_id)
            
            # If no components found, we have a cycle - just take one
            if not group:
                group = [next(iter(remaining))]
            
            # Add group and remove from remaining
            dependency_groups.append(group)
            remaining -= set(group)
        
        # Add dependency groups to plan
        for i, group in enumerate(dependency_groups):
            plan["dependency_groups"][f"group_{i+1}"] = [
                self.knowledge_graph.get_component(component_id)
                for component_id in group
                if self.knowledge_graph.get_component(component_id)
            ]
        
        # Generate implementation steps
        steps = []
        for i, group in enumerate(dependency_groups):
            step = {
                "step": i + 1,
                "description": f"Implement Group {i+1}",
                "components": []
            }
            
            for component_id in group:
                component = self.knowledge_graph.get_component(component_id)
                if component:
                    step["components"].append({
                        "id": component["id"],
                        "name": component["name"],
                        "type": component["type"],
                        "status": component["status"],
                        "description": component["description"]
                    })
            
            steps.append(step)
        
        plan["implementation_steps"] = steps
        
        logger.info(f"Created implementation plan with {len(steps)} steps")
        return plan
    
    def _summarize_component(self, component: Dict[str, Any]) -> str:
        """Create a summary of a component.
        
        Args:
            component: The component to summarize
            
        Returns:
            summary: Summary text
        """
        # Simple summary for now
        name = component.get("name", "Unnamed component")
        component_type = component.get("type", "unknown")
        description = component.get("description", "No description available")
        status = component.get("status", "unknown")
        
        # Additional details based on component type
        if component_type == ComponentType.FUNCTION:
            signature = component.get("metadata", {}).get("signature", name + "()")
            summary = f"Function '{signature}': {description}"
        elif component_type == ComponentType.METHOD:
            signature = component.get("metadata", {}).get("signature", name + "()")
            class_name = component.get("metadata", {}).get("class", "Unknown class")
            summary = f"Method '{class_name}.{signature}': {description}"
        elif component_type == ComponentType.CLASS:
            methods = component.get("metadata", {}).get("methods", [])
            method_list = ", ".join(methods[:3]) + ("..." if len(methods) > 3 else "")
            summary = f"Class '{name}' with methods [{method_list}]: {description}"
        elif component_type == ComponentType.FILE:
            path = component.get("metadata", {}).get("path", name)
            summary = f"File '{path}': {description}"
        else:
            summary = f"{component_type.capitalize()} '{name}': {description}"
        
        return summary
    
    def _simple_similarity(self, str1: str, str2: str) -> float:
        """Calculate a simple similarity score between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            similarity: Similarity score between 0 and 1
        """
        # Calculate Jaccard similarity between words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
