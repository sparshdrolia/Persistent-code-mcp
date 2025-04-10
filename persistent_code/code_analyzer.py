"""
Code Analysis Engine

Responsible for parsing code using Python's AST and extracting entities,
relationships, and metadata.
"""

import ast
import os
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
import networkx as nx

from .knowledge_graph import (
    KnowledgeGraph,
    ComponentType,
    ComponentStatus,
    RelationshipType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeVisitor(ast.NodeVisitor):
    """AST visitor that tracks parent nodes."""
    
    def __init__(self):
        self.parents = {}
    
    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        return super().visit(node)
    
    def get_parent(self, node):
        """Get parent of a node safely."""
        return self.parents.get(node)

class CodeAnalyzer:
    """Analyzes Python code and updates the knowledge graph."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """Initialize the code analyzer.
        
        Args:
            knowledge_graph: The knowledge graph to update
        """
        self.knowledge_graph = knowledge_graph
    
    def analyze_code(self, code_text: str, file_path: Optional[str] = None, component_id: Optional[str] = None) -> str:
        """Analyze code and update the knowledge graph.
        
        Args:
            code_text: The code to analyze
            file_path: Path to the file (optional)
            component_id: ID of an existing component to update (optional)
            
        Returns:
            component_id: ID of the analyzed component
        """
        try:
            # Parse the code
            tree = ast.parse(code_text)
            
            # Process AST to add parent references
            visitor = CodeVisitor()
            visitor.visit(tree)
            
            # File name from path if provided
            file_name = os.path.basename(file_path) if file_path else "unnamed_file.py"
            
            # Determine if this is a new file or an update
            if component_id is None:
                # Create a new file component
                component_id = self.knowledge_graph.add_component(
                    name=file_name,
                    component_type=ComponentType.FILE,
                    code_text=code_text,
                    status=ComponentStatus.IMPLEMENTED,
                    description=f"Python file: {file_name}",
                    metadata={"path": file_path} if file_path else {}
                )
            else:
                # Update existing component
                self.knowledge_graph.update_component(
                    component_id=component_id,
                    code_text=code_text,
                    status=ComponentStatus.IMPLEMENTED
                )
            
            # Extract imports, classes, functions, and variables
            imported_modules = self._analyze_imports(tree, component_id)
            classes_map = self._analyze_classes(tree, component_id, visitor)
            functions_map = self._analyze_functions(tree, component_id, visitor)
            variables_map = self._analyze_variables(tree, component_id, visitor)
            
            # Analyze function calls and other relationships
            self._analyze_relationships(tree, component_id, functions_map, classes_map, visitor)
            
            logger.info(f"Successfully analyzed code from {file_name}")
            return component_id
            
        except SyntaxError as e:
            # If there's a syntax error, still add the component but mark as partial
            logger.warning(f"Syntax error in code: {str(e)}")
            if component_id is None:
                # Create a new file component
                component_id = self.knowledge_graph.add_component(
                    name=os.path.basename(file_path) if file_path else "unnamed_file.py",
                    component_type=ComponentType.FILE,
                    code_text=code_text,
                    status=ComponentStatus.PARTIAL,
                    description=f"Python file with syntax errors: {e}",
                    metadata={"path": file_path, "error": str(e)} if file_path else {"error": str(e)}
                )
            else:
                # Update existing component
                self.knowledge_graph.update_component(
                    component_id=component_id,
                    code_text=code_text,
                    status=ComponentStatus.PARTIAL,
                    metadata={"error": str(e)}
                )
            
            return component_id
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            # Return component_id if we have one, otherwise return None
            return component_id if component_id else None
    
    def _analyze_imports(self, tree: ast.Module, file_id: str) -> List[Tuple[str, str]]:
        """Extract and record imports from the AST.
        
        Args:
            tree: AST of the code
            file_id: ID of the file component
            
        Returns:
            imported_modules: List of (module_name, component_id) tuples
        """
        imported_modules = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Create a module component for the import
                    module_id = self.knowledge_graph.add_component(
                        name=name.name,
                        component_type=ComponentType.MODULE,
                        status=ComponentStatus.IMPLEMENTED,
                        description=f"Imported module: {name.name}",
                        metadata={"external": True, "alias": name.asname}
                    )
                    
                    # Create a relationship between the file and the module
                    self.knowledge_graph.add_relationship(
                        source_id=file_id,
                        target_id=module_id,
                        relationship_type=RelationshipType.IMPORTS
                    )
                    
                    imported_modules.append((name.name, module_id))
                    
            elif isinstance(node, ast.ImportFrom):
                # Handle from X import Y
                module_name = node.module or ""
                
                # Create a module component for the parent module
                parent_module_id = self.knowledge_graph.add_component(
                    name=module_name,
                    component_type=ComponentType.MODULE,
                    status=ComponentStatus.IMPLEMENTED,
                    description=f"Imported module: {module_name}",
                    metadata={"external": True}
                )
                
                # Create relationships for each imported name
                for name in node.names:
                    # Create a component for the imported element
                    import_id = self.knowledge_graph.add_component(
                        name=f"{module_name}.{name.name}" if module_name else name.name,
                        component_type=ComponentType.FUNCTION,  # Assuming function for simplicity
                        status=ComponentStatus.IMPLEMENTED,
                        description=f"Imported from {module_name}: {name.name}",
                        metadata={"external": True, "alias": name.asname, "parent_module": module_name}
                    )
                    
                    # Create relationships
                    self.knowledge_graph.add_relationship(
                        source_id=file_id,
                        target_id=import_id,
                        relationship_type=RelationshipType.IMPORTS
                    )
                    
                    self.knowledge_graph.add_relationship(
                        source_id=parent_module_id,
                        target_id=import_id,
                        relationship_type=RelationshipType.CONTAINS
                    )
                    
                    imported_modules.append((f"{module_name}.{name.name}" if module_name else name.name, import_id))
        
        return imported_modules
    
    def _analyze_classes(self, tree: ast.Module, file_id: str, visitor: CodeVisitor) -> Dict[str, str]:
        """Extract and record classes from the AST.
        
        Args:
            tree: AST of the code
            file_id: ID of the file component
            visitor: AST visitor with parent tracking
            
        Returns:
            class_map: Dictionary mapping class names to component IDs
        """
        class_map = {}
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # Extract class docstring
                docstring = ast.get_docstring(node) or ""
                
                # Extract base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                
                # Extract class code
                class_code = ""
                try:
                    # Get line numbers
                    start_line = node.lineno
                    end_line = 0
                    
                    # Try to get end line number
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        end_line = node.end_lineno
                    else:
                        # Approximate using the last child
                        last_child = node.body[-1] if node.body else node
                        if hasattr(last_child, 'end_lineno') and last_child.end_lineno:
                            end_line = last_child.end_lineno
                        else:
                            end_line = start_line + 5  # Rough estimate
                    
                    # Get the source code for the file
                    code_lines = self.knowledge_graph.get_component(file_id)["code_text"].split('\n')
                    
                    # Extract class code
                    if start_line <= end_line and start_line <= len(code_lines):
                        class_code = '\n'.join(code_lines[start_line-1:end_line])
                except Exception as e:
                    logger.warning(f"Error extracting class code for {node.name}: {str(e)}")
                    class_code = f"class {node.name}(...): ..."
                
                # Create class component
                class_id = self.knowledge_graph.add_component(
                    name=node.name,
                    component_type=ComponentType.CLASS,
                    code_text=class_code,
                    status=ComponentStatus.IMPLEMENTED,
                    description=docstring,
                    metadata={
                        "bases": bases,
                        "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        "line_range": (node.lineno, end_line)
                    }
                )
                
                # Add relationship between file and class
                self.knowledge_graph.add_relationship(
                    source_id=file_id,
                    target_id=class_id,
                    relationship_type=RelationshipType.CONTAINS
                )
                
                # Add inheritance relationships
                for base in bases:
                    # Search for base class in the graph
                    base_components = self.knowledge_graph.search_code(
                        query=base,
                        component_types=[ComponentType.CLASS],
                        limit=1
                    )
                    
                    if base_components:
                        base_id = base_components[0]["id"]
                        self.knowledge_graph.add_relationship(
                            source_id=class_id,
                            target_id=base_id,
                            relationship_type=RelationshipType.INHERITS
                        )
                
                # Analyze methods within the class
                method_map = self._analyze_methods(node, class_id, visitor)
                
                # Add class to map
                class_map[node.name] = class_id
        
        return class_map
    
    def _analyze_methods(self, class_node: ast.ClassDef, class_id: str, visitor: CodeVisitor) -> Dict[str, str]:
        """Extract and record methods from a class definition.
        
        Args:
            class_node: AST node for the class
            class_id: ID of the class component
            visitor: AST visitor with parent tracking
            
        Returns:
            method_map: Dictionary mapping method names to component IDs
        """
        method_map = {}
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                # Extract method docstring
                docstring = ast.get_docstring(node) or ""
                
                # Extract method signature
                arg_names = [a.arg for a in node.args.args]
                signature = f"{node.name}({', '.join(arg_names)})"
                
                # Extract method code
                method_code = ""
                try:
                    # Get the source code for the class
                    class_code = self.knowledge_graph.get_component(class_id)["code_text"].split('\n')
                    
                    # Get line numbers relative to the class
                    start_line = node.lineno - class_node.lineno
                    end_line = 0
                    
                    # Try to get end line number
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        end_line = node.end_lineno - class_node.lineno
                    else:
                        # Approximate using the last child
                        last_child = node.body[-1] if node.body else node
                        if hasattr(last_child, 'end_lineno') and last_child.end_lineno:
                            end_line = last_child.end_lineno - class_node.lineno
                        else:
                            end_line = start_line + 5  # Rough estimate
                    
                    # Extract method code
                    if 0 <= start_line <= end_line and start_line < len(class_code):
                        method_code = '\n'.join(class_code[start_line:end_line+1])
                    else:
                        method_code = f"def {signature}: ..."
                except Exception as e:
                    logger.warning(f"Error extracting method code for {class_node.name}.{node.name}: {str(e)}")
                    method_code = f"def {signature}: ..."
                
                # Create method component
                method_id = self.knowledge_graph.add_component(
                    name=f"{class_node.name}.{node.name}",
                    component_type=ComponentType.METHOD,
                    code_text=method_code,
                    status=ComponentStatus.IMPLEMENTED,
                    description=docstring,
                    metadata={
                        "signature": signature,
                        "class": class_node.name,
                        "is_static": any(d.id == 'staticmethod' for d in node.decorator_list if isinstance(d, ast.Name)),
                        "is_class_method": any(d.id == 'classmethod' for d in node.decorator_list if isinstance(d, ast.Name)),
                        "args": arg_names,
                        "line_range": (node.lineno, getattr(node, 'end_lineno', 0))
                    }
                )
                
                # Add relationship between class and method
                self.knowledge_graph.add_relationship(
                    source_id=class_id,
                    target_id=method_id,
                    relationship_type=RelationshipType.CONTAINS
                )
                
                # Add method to map
                method_map[node.name] = method_id
        
        return method_map
    
    def _analyze_functions(self, tree: ast.Module, file_id: str, visitor: CodeVisitor) -> Dict[str, str]:
        """Extract and record functions from the AST.
        
        Args:
            tree: AST of the code
            file_id: ID of the file component
            visitor: AST visitor with parent tracking
            
        Returns:
            function_map: Dictionary mapping function names to component IDs
        """
        function_map = {}
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip methods (functions inside classes)
                parent = visitor.get_parent(node)
                if parent and isinstance(parent, ast.ClassDef):
                    continue
                
                # Extract function docstring
                docstring = ast.get_docstring(node) or ""
                
                # Extract function signature
                arg_names = [a.arg for a in node.args.args]
                signature = f"{node.name}({', '.join(arg_names)})"
                
                # Extract function code
                function_code = ""
                try:
                    # Get line numbers
                    start_line = node.lineno
                    end_line = 0
                    
                    # Try to get end line number
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        end_line = node.end_lineno
                    else:
                        # Approximate using the last child
                        last_child = node.body[-1] if node.body else node
                        if hasattr(last_child, 'end_lineno') and last_child.end_lineno:
                            end_line = last_child.end_lineno
                        else:
                            end_line = start_line + 5  # Rough estimate
                    
                    # Get the source code for the file
                    code_lines = self.knowledge_graph.get_component(file_id)["code_text"].split('\n')
                    
                    # Extract function code
                    if start_line <= end_line and start_line <= len(code_lines):
                        function_code = '\n'.join(code_lines[start_line-1:end_line])
                except Exception as e:
                    logger.warning(f"Error extracting function code for {node.name}: {str(e)}")
                    function_code = f"def {signature}: ..."
                
                # Create function component
                function_id = self.knowledge_graph.add_component(
                    name=node.name,
                    component_type=ComponentType.FUNCTION,
                    code_text=function_code,
                    status=ComponentStatus.IMPLEMENTED,
                    description=docstring,
                    metadata={
                        "signature": signature,
                        "args": arg_names,
                        "line_range": (node.lineno, end_line)
                    }
                )
                
                # Add relationship between file and function
                self.knowledge_graph.add_relationship(
                    source_id=file_id,
                    target_id=function_id,
                    relationship_type=RelationshipType.CONTAINS
                )
                
                # Add function to map
                function_map[node.name] = function_id
        
        return function_map
    
    def _analyze_variables(self, tree: ast.Module, file_id: str, visitor: CodeVisitor) -> Dict[str, str]:
        """Extract and record global variables from the AST.
        
        Args:
            tree: AST of the code
            file_id: ID of the file component
            visitor: AST visitor with parent tracking
            
        Returns:
            variable_map: Dictionary mapping variable names to component IDs
        """
        variable_map = {}
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                # Skip variables inside functions or classes
                parent = visitor.get_parent(node)
                if parent and (isinstance(parent, ast.ClassDef) or isinstance(parent, ast.FunctionDef)):
                    continue
                
                # Extract variable names
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Create variable component
                        variable_id = self.knowledge_graph.add_component(
                            name=target.id,
                            component_type=ComponentType.VARIABLE,
                            code_text=f"{target.id} = ...",
                            status=ComponentStatus.IMPLEMENTED,
                            description=f"Global variable: {target.id}",
                            metadata={
                                "line": node.lineno
                            }
                        )
                        
                        # Add relationship between file and variable
                        self.knowledge_graph.add_relationship(
                            source_id=file_id,
                            target_id=variable_id,
                            relationship_type=RelationshipType.CONTAINS
                        )
                        
                        # Add variable to map
                        variable_map[target.id] = variable_id
        
        return variable_map
    
    def _analyze_relationships(self, 
                            tree: ast.Module, 
                            file_id: str, 
                            functions: Dict[str, str],
                            classes: Dict[str, str],
                            visitor: CodeVisitor):
        """Analyze function calls and other relationships from the AST.
        
        Args:
            tree: AST of the code
            file_id: ID of the file component
            functions: Dictionary of function names to component IDs
            classes: Dictionary of class names to component IDs
            visitor: AST visitor with parent tracking
        """
        # Use custom visitor to find all function calls
        call_visitor = FunctionCallVisitor(self, file_id, functions, classes, visitor)
        call_visitor.visit(tree)

class FunctionCallVisitor(ast.NodeVisitor):
    """Visitor that finds function calls and other relationships."""
    
    def __init__(self, analyzer, file_id, functions, classes, parent_visitor):
        self.analyzer = analyzer
        self.file_id = file_id
        self.functions = functions
        self.classes = classes
        self.parent_visitor = parent_visitor
        self.current_function = None
        self.current_method = None
        self.calls = []
    
    def visit_FunctionDef(self, node):
        # Track current function for context
        old_function = self.current_function
        old_method = self.current_method
        
        # Check if this is a method or a function
        parent = self.parent_visitor.get_parent(node)
        if parent and isinstance(parent, ast.ClassDef):
            self.current_method = f"{parent.name}.{node.name}"
            self.current_function = None
        else:
            self.current_function = node.name
            self.current_method = None
        
        # Visit children
        self.generic_visit(node)
        
        # Restore context
        self.current_function = old_function
        self.current_method = old_method
    
    def visit_Call(self, node):
        # Extract function name
        func_name = None
        
        if isinstance(node.func, ast.Name):
            # Direct function call: func()
            func_name = node.func.id
            
            # Check if this is a known function
            if func_name in self.functions:
                caller_id = None
                
                # Get caller ID
                if self.current_function and self.current_function in self.functions:
                    caller_id = self.functions[self.current_function]
                elif self.current_method:
                    # Find method component
                    method_components = self.analyzer.knowledge_graph.search_code(
                        query=self.current_method,
                        component_types=[ComponentType.METHOD],
                        limit=1
                    )
                    if method_components:
                        caller_id = method_components[0]["id"]
                
                if caller_id:
                    # Add relationship for the function call
                    self.analyzer.knowledge_graph.add_relationship(
                        source_id=caller_id,
                        target_id=self.functions[func_name],
                        relationship_type=RelationshipType.CALLS
                    )
        
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                
                # Check if this is a known class
                if obj_name in self.classes:
                    # Find method component
                    method_components = self.analyzer.knowledge_graph.search_code(
                        query=f"{obj_name}.{method_name}",
                        component_types=[ComponentType.METHOD],
                        limit=1
                    )
                    
                    if method_components:
                        caller_id = None
                        
                        # Get caller ID
                        if self.current_function and self.current_function in self.functions:
                            caller_id = self.functions[self.current_function]
                        elif self.current_method:
                            # Find method component
                            current_method_components = self.analyzer.knowledge_graph.search_code(
                                query=self.current_method,
                                component_types=[ComponentType.METHOD],
                                limit=1
                            )
                            if current_method_components:
                                caller_id = current_method_components[0]["id"]
                        
                        if caller_id:
                            # Add relationship for the method call
                            self.analyzer.knowledge_graph.add_relationship(
                                source_id=caller_id,
                                target_id=method_components[0]["id"],
                                relationship_type=RelationshipType.CALLS
                            )
        
        # Visit children
        self.generic_visit(node)
