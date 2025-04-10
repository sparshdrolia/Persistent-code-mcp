"""
MCP Server Interface

Implements the Model Context Protocol (MCP) server for the Persistent-Code tool.
Exposes the knowledge graph functionality through well-defined tools.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .knowledge_graph import (
    KnowledgeGraph,
    ComponentType,
    ComponentStatus,
    RelationshipType
)
from .code_analyzer import CodeAnalyzer
from .context_assembler import ContextAssembler

class PersistentCodeMCP:
    """MCP server for Persistent-Code knowledge graph."""
    
    def __init__(self, project_name: str = "default", storage_dir: str = None):
        """Initialize the MCP server.
        
        Args:
            project_name: Name of the project
            storage_dir: Directory to store persistent data
        """
        self.project_name = project_name
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "storage")
        
        # Initialize MCP server
        self.mcp = FastMCP("persistent-code")
        
        # Initialize components
        self.knowledge_graph = KnowledgeGraph(project_name, storage_dir)
        self.code_analyzer = CodeAnalyzer(self.knowledge_graph)
        self.context_assembler = ContextAssembler(self.knowledge_graph)
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all MCP tools."""
        # Knowledge Graph Management tools
        self.mcp.tool()(self.add_component)
        self.mcp.tool()(self.update_component)
        self.mcp.tool()(self.add_relationship)
        
        # Code Retrieval and Navigation tools
        self.mcp.tool()(self.get_component)
        self.mcp.tool()(self.find_related_components)
        self.mcp.tool()(self.search_code)
        
        # Status Management tools
        self.mcp.tool()(self.update_status)
        self.mcp.tool()(self.get_project_status)
        self.mcp.tool()(self.find_next_tasks)
        
        # Context Assembly tools
        self.mcp.tool()(self.prepare_context)
        self.mcp.tool()(self.continue_implementation)
        self.mcp.tool()(self.get_implementation_plan)
        
        # Code Analysis tools
        self.mcp.tool()(self.analyze_code)
    
    def run(self, transport: str = 'stdio'):
        """Run the MCP server.
        
        Args:
            transport: Transport protocol ('stdio' or 'http')
        """
        self.mcp.run(transport=transport)
    
    # ==========================================================================
    # Knowledge Graph Management tools
    # ==========================================================================
    
    async def add_component(self, 
                          code_text: str, 
                          component_type: str, 
                          name: str, 
                          status: str = "planned", 
                          description: str = "") -> Dict[str, Any]:
        """Add a new code component to the graph.
        
        Args:
            code_text: The actual code
            component_type: Type of component (file, class, function, method, variable)
            name: Name of the component
            status: Implementation status (planned, partial, implemented)
            description: Semantic description of the component
            
        Returns:
            result: Result including component_id
        """
        try:
            # Convert string enums to actual enums
            comp_type = ComponentType(component_type)
            comp_status = ComponentStatus(status)
            
            # Add the component
            component_id = self.knowledge_graph.add_component(
                name=name,
                component_type=comp_type,
                code_text=code_text,
                status=comp_status,
                description=description
            )
            
            return {
                "success": True,
                "component_id": component_id,
                "message": f"Added {component_type} '{name}' with ID {component_id}"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_component(self, 
                             component_id: str, 
                             code_text: Optional[str] = None, 
                             status: Optional[str] = None, 
                             description: Optional[str] = None) -> Dict[str, Any]:
        """Update an existing component.
        
        Args:
            component_id: ID of the component to update
            code_text: New code text (if changed)
            status: New implementation status (if changed)
            description: New description (if changed)
            
        Returns:
            result: Success status
        """
        try:
            # Convert string enum to actual enum if provided
            comp_status = None
            if status:
                comp_status = ComponentStatus(status)
            
            # Update the component
            success = self.knowledge_graph.update_component(
                component_id=component_id,
                code_text=code_text,
                status=comp_status,
                description=description
            )
            
            if success:
                return {
                    "success": True,
                    "message": f"Updated component {component_id}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Component {component_id} not found"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_relationship(self, 
                             source_id: str, 
                             target_id: str, 
                             relationship_type: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a relationship between components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            relationship_type: Type of relationship (imports, calls, inherits, contains, depends_on)
            metadata: Additional metadata for the relationship
            
        Returns:
            result: Result including relationship_id
        """
        try:
            # Convert string enum to actual enum
            rel_type = RelationshipType(relationship_type)
            
            # Add the relationship
            relationship_id = self.knowledge_graph.add_relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                metadata=metadata
            )
            
            if relationship_id:
                return {
                    "success": True,
                    "relationship_id": relationship_id,
                    "message": f"Added {relationship_type} relationship from {source_id} to {target_id}"
                }
            else:
                return {
                    "success": False,
                    "error": "One or both components not found"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==========================================================================
    # Code Retrieval and Navigation tools
    # ==========================================================================
    
    async def get_component(self, 
                          identifier: str, 
                          include_related: bool = False) -> Dict[str, Any]:
        """Retrieve a component by ID or name.
        
        Args:
            identifier: Component ID or name
            include_related: Whether to include related components
            
        Returns:
            result: Component details
        """
        try:
            # Try as ID first
            component = self.knowledge_graph.get_component(identifier, include_related)
            
            if component:
                return {
                    "success": True,
                    "component": component
                }
            
            # If not found, try searching by name
            search_results = self.knowledge_graph.search_code(
                query=identifier,
                limit=1
            )
            
            if search_results:
                component = self.knowledge_graph.get_component(
                    search_results[0]["id"],
                    include_related
                )
                return {
                    "success": True,
                    "component": component
                }
            
            return {
                "success": False,
                "error": f"Component '{identifier}' not found"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def find_related_components(self, 
                                    component_id: str, 
                                    relationship_types: Optional[List[str]] = None, 
                                    depth: int = 1) -> Dict[str, Any]:
        """Find components related to a given component.
        
        Args:
            component_id: ID of the component
            relationship_types: Types of relationships to consider
            depth: How many levels of relationships to traverse
            
        Returns:
            result: List of related components
        """
        try:
            # Convert string enums to actual enums if provided
            rel_types = None
            if relationship_types:
                rel_types = [RelationshipType(rt) for rt in relationship_types]
            
            # Find related components
            related = self.knowledge_graph.find_related_components(
                component_id=component_id,
                relationship_types=rel_types,
                depth=depth
            )
            
            return {
                "success": True,
                "related_components": related,
                "count": len(related)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_code(self, 
                        query: str, 
                        component_types: Optional[List[str]] = None, 
                        limit: int = 10) -> Dict[str, Any]:
        """Search the codebase semantically.
        
        Args:
            query: Search query
            component_types: Types of components to search
            limit: Maximum number of results
            
        Returns:
            result: Ranked list of matching components
        """
        try:
            # Convert string enums to actual enums if provided
            comp_types = None
            if component_types:
                comp_types = [ComponentType(ct) for ct in component_types]
            
            # Search the code
            matches = self.knowledge_graph.search_code(
                query=query,
                component_types=comp_types,
                limit=limit
            )
            
            return {
                "success": True,
                "matches": matches,
                "count": len(matches)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==========================================================================
    # Status Management tools
    # ==========================================================================
    
    async def update_status(self, 
                          component_id: str, 
                          new_status: str, 
                          notes: str = "") -> Dict[str, Any]:
        """Update implementation status of a component.
        
        Args:
            component_id: ID of the component
            new_status: New implementation status (planned, partial, implemented)
            notes: Optional notes about the status change
            
        Returns:
            result: Update confirmation
        """
        try:
            # Convert string enum to actual enum
            status = ComponentStatus(new_status)
            
            # Update the status
            success = self.knowledge_graph.update_status(
                component_id=component_id,
                new_status=status,
                notes=notes
            )
            
            if success:
                return {
                    "success": True,
                    "message": f"Updated status of {component_id} to {new_status}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Component {component_id} not found"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_project_status(self, 
                               filters: Optional[Dict[str, Any]] = None, 
                               grouping: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve implementation status across the project.
        
        Args:
            filters: Filters to apply to components
            grouping: How to group the results (e.g., by type)
            
        Returns:
            result: Status summary
        """
        try:
            # Get project status
            status = self.knowledge_graph.get_project_status(
                filters=filters,
                grouping=grouping
            )
            
            return {
                "success": True,
                "status": status
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def find_next_tasks(self, 
                            priority_type: str = "dependencies", 
                            limit: int = 5) -> Dict[str, Any]:
        """Suggest logical next components to implement.
        
        Args:
            priority_type: How to prioritize tasks (dependencies, complexity)
            limit: Maximum number of tasks to suggest
            
        Returns:
            result: List of suggested tasks
        """
        try:
            # Find next tasks
            tasks = self.knowledge_graph.find_next_tasks(
                priority_type=priority_type,
                limit=limit
            )
            
            return {
                "success": True,
                "tasks": tasks,
                "count": len(tasks)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==========================================================================
    # Context Assembly tools
    # ==========================================================================
    
    async def prepare_context(self, 
                            task_description: str, 
                            relevant_components: Optional[List[str]] = None, 
                            max_tokens: int = 4000) -> Dict[str, Any]:
        """Assemble minimal context for a specific task.
        
        Args:
            task_description: Description of the task
            relevant_components: List of component IDs known to be relevant
            max_tokens: Maximum tokens for the context
            
        Returns:
            result: Assembled context
        """
        try:
            # Prepare context
            context = self.context_assembler.prepare_context(
                task_description=task_description,
                relevant_components=relevant_components,
                max_tokens=max_tokens
            )
            
            return {
                "success": True,
                "context": context,
                "primary_count": len(context["primary_components"]),
                "related_count": len(context["related_components"]),
                "summary_count": len(context["summaries"])
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def continue_implementation(self, 
                                    component_id: str, 
                                    max_tokens: int = 4000) -> Dict[str, Any]:
        """Provide context to continue implementing a component.
        
        Args:
            component_id: ID of the component to continue implementing
            max_tokens: Maximum tokens for the context
            
        Returns:
            result: Implementation context
        """
        try:
            # Get implementation context
            context = self.context_assembler.continue_implementation(
                component_id=component_id,
                max_tokens=max_tokens
            )
            
            if "error" in context:
                return {
                    "success": False,
                    "error": context["error"]
                }
            
            return {
                "success": True,
                "context": context
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_implementation_plan(self, 
                                    component_ids: Optional[List[str]] = None, 
                                    dependencies_depth: int = 1) -> Dict[str, Any]:
        """Generate a plan for implementing pending components.
        
        Args:
            component_ids: List of component IDs to include
            dependencies_depth: How deep to analyze dependencies
            
        Returns:
            result: Ordered implementation plan
        """
        try:
            # Get implementation plan
            plan = self.context_assembler.get_implementation_plan(
                component_ids=component_ids,
                dependencies_depth=dependencies_depth
            )
            
            if "error" in plan:
                return {
                    "success": False,
                    "error": plan["error"]
                }
            
            return {
                "success": True,
                "plan": plan
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==========================================================================
    # Code Analysis tools
    # ==========================================================================
    
    async def analyze_code(self, 
                         code_text: str, 
                         file_path: Optional[str] = None, 
                         component_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze code and update the knowledge graph.
        
        Args:
            code_text: The code to analyze
            file_path: Path to the file (optional)
            component_id: ID of an existing component to update (optional)
            
        Returns:
            result: Analysis result
        """
        try:
            # Analyze the code
            component_id = self.code_analyzer.analyze_code(
                code_text=code_text,
                file_path=file_path,
                component_id=component_id
            )
            
            # Get the updated component
            component = self.knowledge_graph.get_component(component_id, include_related=True)
            
            return {
                "success": True,
                "component_id": component_id,
                "component": component,
                "message": f"Successfully analyzed and updated knowledge graph"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
