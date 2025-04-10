"""
Tests for the knowledge graph module.
"""

import os
import tempfile
import unittest
import shutil
from pathlib import Path

from persistent_code.knowledge_graph import (
    KnowledgeGraph,
    ComponentType,
    ComponentStatus,
    RelationshipType
)

class TestKnowledgeGraph(unittest.TestCase):
    """Test cases for the KnowledgeGraph class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.graph = KnowledgeGraph("test", storage_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_add_component(self):
        """Test adding a component to the graph."""
        # Add a component
        component_id = self.graph.add_component(
            name="TestComponent",
            component_type=ComponentType.FILE,
            code_text="print('Hello, World!')",
            status=ComponentStatus.IMPLEMENTED,
            description="A test component"
        )
        
        # Check that the component was added
        self.assertIn(component_id, self.graph.graph)
        
        # Check the component data
        component = self.graph.graph.nodes[component_id]
        self.assertEqual(component["name"], "TestComponent")
        self.assertEqual(component["type"], ComponentType.FILE)
        self.assertEqual(component["code_text"], "print('Hello, World!')")
        self.assertEqual(component["status"], ComponentStatus.IMPLEMENTED)
        self.assertEqual(component["description"], "A test component")
    
    def test_update_component(self):
        """Test updating a component in the graph."""
        # Add a component
        component_id = self.graph.add_component(
            name="TestComponent",
            component_type=ComponentType.FILE,
            code_text="print('Hello, World!')",
            status=ComponentStatus.IMPLEMENTED,
            description="A test component"
        )
        
        # Update the component
        self.graph.update_component(
            component_id=component_id,
            code_text="print('Updated!')",
            status=ComponentStatus.PARTIAL,
            description="An updated test component"
        )
        
        # Check the updated component data
        component = self.graph.graph.nodes[component_id]
        self.assertEqual(component["name"], "TestComponent")  # Name should not change
        self.assertEqual(component["type"], ComponentType.FILE)  # Type should not change
        self.assertEqual(component["code_text"], "print('Updated!')")
        self.assertEqual(component["status"], ComponentStatus.PARTIAL)
        self.assertEqual(component["description"], "An updated test component")
    
    def test_add_relationship(self):
        """Test adding a relationship between components."""
        # Add two components
        source_id = self.graph.add_component(
            name="SourceComponent",
            component_type=ComponentType.CLASS,
            status=ComponentStatus.IMPLEMENTED
        )
        
        target_id = self.graph.add_component(
            name="TargetComponent",
            component_type=ComponentType.METHOD,
            status=ComponentStatus.IMPLEMENTED
        )
        
        # Add a relationship
        relationship_id = self.graph.add_relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=RelationshipType.CONTAINS
        )
        
        # Check that the relationship was added
        self.assertIn(target_id, self.graph.graph[source_id])
        
        # Check the relationship data
        edge_data = self.graph.graph[source_id][target_id]
        self.assertEqual(edge_data["type"], RelationshipType.CONTAINS)
    
    def test_get_component(self):
        """Test retrieving a component from the graph."""
        # Add a component
        component_id = self.graph.add_component(
            name="TestComponent",
            component_type=ComponentType.FILE,
            code_text="print('Hello, World!')",
            status=ComponentStatus.IMPLEMENTED,
            description="A test component"
        )
        
        # Get the component
        component = self.graph.get_component(component_id)
        
        # Check the component data
        self.assertEqual(component["name"], "TestComponent")
        self.assertEqual(component["type"], ComponentType.FILE)
        self.assertEqual(component["code_text"], "print('Hello, World!')")
        self.assertEqual(component["status"], ComponentStatus.IMPLEMENTED)
        self.assertEqual(component["description"], "A test component")
    
    def test_find_related_components(self):
        """Test finding related components."""
        # Add three components
        class_id = self.graph.add_component(
            name="TestClass",
            component_type=ComponentType.CLASS,
            status=ComponentStatus.IMPLEMENTED
        )
        
        method1_id = self.graph.add_component(
            name="method1",
            component_type=ComponentType.METHOD,
            status=ComponentStatus.IMPLEMENTED
        )
        
        method2_id = self.graph.add_component(
            name="method2",
            component_type=ComponentType.METHOD,
            status=ComponentStatus.IMPLEMENTED
        )
        
        # Add relationships
        self.graph.add_relationship(
            source_id=class_id,
            target_id=method1_id,
            relationship_type=RelationshipType.CONTAINS
        )
        
        self.graph.add_relationship(
            source_id=class_id,
            target_id=method2_id,
            relationship_type=RelationshipType.CONTAINS
        )
        
        # Find related components
        related = self.graph.find_related_components(
            component_id=class_id,
            relationship_types=[RelationshipType.CONTAINS],
            depth=1
        )
        
        # Check the related components
        self.assertEqual(len(related), 2)
        related_ids = {r["id"] for r in related}
        self.assertIn(method1_id, related_ids)
        self.assertIn(method2_id, related_ids)
    
    def test_search_code(self):
        """Test searching for code."""
        # Add components with different names and descriptions
        self.graph.add_component(
            name="UserService",
            component_type=ComponentType.CLASS,
            description="Service for managing users",
            status=ComponentStatus.IMPLEMENTED
        )
        
        self.graph.add_component(
            name="ProductService",
            component_type=ComponentType.CLASS,
            description="Service for managing products",
            status=ComponentStatus.IMPLEMENTED
        )
        
        self.graph.add_component(
            name="AuthController",
            component_type=ComponentType.CLASS,
            description="Controller for authentication",
            status=ComponentStatus.IMPLEMENTED
        )
        
        # Search for "user"
        results = self.graph.search_code(query="user")
        
        # Check the search results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "UserService")
    
    def test_save_and_load(self):
        """Test saving and loading the graph."""
        # Add a component
        component_id = self.graph.add_component(
            name="TestComponent",
            component_type=ComponentType.FILE,
            code_text="print('Hello, World!')",
            status=ComponentStatus.IMPLEMENTED,
            description="A test component"
        )
        
        # Save the graph
        self.graph.save()
        
        # Create a new graph instance
        new_graph = KnowledgeGraph("test", storage_dir=self.test_dir)
        
        # Check that the component was loaded
        self.assertIn(component_id, new_graph.graph)
        
        # Check the component data
        component = new_graph.graph.nodes[component_id]
        self.assertEqual(component["name"], "TestComponent")
        self.assertEqual(component["type"], ComponentType.FILE)
        self.assertEqual(component["code_text"], "print('Hello, World!')")
        self.assertEqual(component["status"], ComponentStatus.IMPLEMENTED)
        self.assertEqual(component["description"], "A test component")

if __name__ == "__main__":
    unittest.main()
