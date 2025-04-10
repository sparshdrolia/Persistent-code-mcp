"""
Demo script for the enhanced Persistent-Code MCP server with LlamaIndex.

This script demonstrates how to use the Persistent-Code MCP server with 
LlamaIndex integration to analyze Python code and perform semantic searches.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import persistent_code
sys.path.insert(0, str(Path(__file__).parent.parent))

from persistent_code.knowledge_graph import (
    KnowledgeGraph, 
    ComponentType, 
    ComponentStatus,
    RelationshipType
)
from persistent_code.code_analyzer import CodeAnalyzer
from persistent_code.context_assembler import ContextAssembler

# Sample code for demo
CALENDAR_APP_CODE = """
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Union

@dataclass
class Calendar:
    """A calendar that can contain events."""
    name: str
    description: str
    events: List["Event"] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
    
    def add_event(self, event: "Event") -> None:
        """Add an event to the calendar."""
        self.events.append(event)
    
    def get_events_on_date(self, date: datetime) -> List["Event"]:
        """Get all events occurring on a specific date."""
        return [
            event for event in self.events 
            if event.starts_on_date(date)
        ]
    
    def find_events_by_title(self, title: str) -> List["Event"]:
        """Find events by title (case-insensitive partial match)."""
        title = title.lower()
        return [
            event for event in self.events
            if title in event.title.lower()
        ]

@dataclass
class Event:
    """An event with a start time and duration."""
    title: str
    description: str
    start_time: datetime
    duration: timedelta
    location: Optional[str] = None
    attendees: List[str] = None
    
    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []
    
    @property
    def end_time(self) -> datetime:
        """Calculate the end time based on start time and duration."""
        return self.start_time + self.duration
    
    def starts_on_date(self, date: datetime) -> bool:
        """Check if the event starts on the given date."""
        return (
            self.start_time.year == date.year and
            self.start_time.month == date.month and
            self.start_time.day == date.day
        )
    
    def add_attendee(self, attendee: str) -> None:
        """Add an attendee to the event."""
        if attendee not in self.attendees:
            self.attendees.append(attendee)
    
    def remove_attendee(self, attendee: str) -> bool:
        """Remove an attendee from the event."""
        if attendee in self.attendees:
            self.attendees.remove(attendee)
            return True
        return False
    
    def is_attendee(self, attendee: str) -> bool:
        """Check if someone is an attendee of the event."""
        return attendee in self.attendees

class CalendarManager:
    """Manages multiple calendars."""
    
    def __init__(self):
        self.calendars: Dict[str, Calendar] = {}
    
    def create_calendar(self, name: str, description: str) -> Calendar:
        """Create a new calendar."""
        calendar = Calendar(name=name, description=description)
        self.calendars[name] = calendar
        return calendar
    
    def get_calendar(self, name: str) -> Optional[Calendar]:
        """Get a calendar by name."""
        return self.calendars.get(name)
    
    def add_event(self, calendar_name: str, event: Event) -> bool:
        """Add an event to a calendar."""
        calendar = self.get_calendar(calendar_name)
        if calendar:
            calendar.add_event(event)
            return True
        return False
    
    def find_events_across_calendars(self, query: str) -> Dict[str, List[Event]]:
        """Find events matching a query across all calendars."""
        results = {}
        for name, calendar in self.calendars.items():
            events = calendar.find_events_by_title(query)
            if events:
                results[name] = events
        return results
    
    def get_events_on_date(self, date: datetime) -> Dict[str, List[Event]]:
        """Get all events on a specific date across all calendars."""
        results = {}
        for name, calendar in self.calendars.items():
            events = calendar.get_events_on_date(date)
            if events:
                results[name] = events
        return results
"""

def main():
    """Run the demo."""
    # Create a project directory
    project_dir = Path("demo_llama_index_project")
    project_dir.mkdir(exist_ok=True)
    
    # Create a knowledge graph with LlamaIndex integration
    print("Creating LlamaIndex-powered knowledge graph...")
    graph = KnowledgeGraph("llama_demo", storage_dir=str(project_dir))
    
    # Create a code analyzer
    analyzer = CodeAnalyzer(graph)
    
    # Analyze the Calendar app code
    print("\nAnalyzing Calendar app code...")
    calendar_file_id = analyzer.analyze_code(
        code_text=CALENDAR_APP_CODE,
        file_path="calendar_app.py"
    )
    
    # Print information about components
    print("\nComponents in the knowledge graph:")
    components = {}
    for node_id, data in graph.graph.nodes(data=True):
        name = data.get("name", "Unknown")
        comp_type = data.get("type", "Unknown")
        components[node_id] = (name, comp_type)
        print(f"- [{comp_type}] {name} (ID: {node_id})")
    
    # Create context assembler
    context_assembler = ContextAssembler(graph)
    
    # Semantic search demo
    print("\nPerforming semantic search for 'find events in calendar'...")
    search_results = graph.search_code(
        query="find events in calendar",
        limit=3
    )
    
    if search_results:
        print("\nSearch results:")
        for result in search_results:
            print(f"- {result['name']} ({result['type']}): {result['description']}")
    
    # Prepare context for implementing a new feature
    print("\nPreparing context for implementing 'recurring events' feature...")
    context = context_assembler.prepare_context(
        task_description="Implement a RecurringEvent class that inherits from Event and supports daily, weekly, and monthly recurrence patterns",
        max_tokens=2000
    )
    
    # Print the primary components in the context
    print("\nPrimary components for the task:")
    for component in context["primary_components"]:
        name = component.get("name", "Unknown")
        comp_type = component.get("type", "Unknown")
        desc = component.get("description", "")[:50] + "..." if len(component.get("description", "")) > 50 else component.get("description", "")
        print(f"- [{comp_type}] {name}: {desc}")
    
    # Generate an implementation plan
    print("\nGenerating implementation plan...")
    plan = context_assembler.get_implementation_plan()
    
    # Check if we have an actual plan or an error
    if "error" in plan:
        print(f"Implementation plan error: {plan['error']}")
        plan_steps = []
    else:
        # Print implementation steps
        plan_steps = plan.get("implementation_steps", [])
    
    if plan_steps:
        print("\nImplementation steps:")
        for step in plan_steps:
            print(f"Step {step['step']}: {step['description']}")
            for component in step["components"]:
                print(f"  - {component['name']} ({component['type']})")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
