"""
Semantic Search Demonstration for Persistent-Code MCP

This script demonstrates the semantic search capabilities of the 
Persistent-Code MCP server with LlamaIndex integration.
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
from persistent_code.config import config

# Sample code for data processing library
DATA_PROCESSING_CODE = """
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormat(str, Enum):
    """Supported data formats for import/export."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PARQUET = "parquet"
    SQL = "sql"

@dataclass
class ColumnInfo:
    """Information about a single column in a dataset."""
    name: str
    dtype: str
    nullable: bool = True
    description: str = ""
    
    @property
    def is_numeric(self) -> bool:
        """Check if the column is numeric."""
        return self.dtype in ('int', 'float', 'Int64', 'Float64', 'int64', 'float64')
    
    @property
    def is_temporal(self) -> bool:
        """Check if the column is a date or time."""
        return self.dtype in ('datetime', 'datetime64', 'date', 'time')

class DataProcessor:
    """Process and transform datasets."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the data processor.
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.logger = logger
        if verbose:
            self.logger.setLevel(logging.DEBUG)
    
    def load_data(self, path: str, format: DataFormat = None) -> pd.DataFrame:
        """Load data from a file.
        
        Args:
            path: Path to the data file
            format: File format (guessed from extension if None)
            
        Returns:
            Loaded dataframe
        """
        if format is None:
            # Guess format from file extension
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            try:
                format = DataFormat(ext)
            except ValueError:
                raise ValueError(f"Unsupported file extension: {ext}")
        
        self.logger.info(f"Loading data from {path} as {format}")
        
        if format == DataFormat.CSV:
            return pd.read_csv(path)
        elif format == DataFormat.EXCEL:
            return pd.read_excel(path)
        elif format == DataFormat.JSON:
            return pd.read_json(path)
        elif format == DataFormat.PARQUET:
            return pd.read_parquet(path)
        elif format == DataFormat.SQL:
            # This would require additional parameters in practice
            raise NotImplementedError("SQL loading not implemented yet")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_data(self, df: pd.DataFrame, path: str, format: DataFormat = None) -> None:
        """Save data to a file.
        
        Args:
            df: DataFrame to save
            path: Path to save to
            format: File format (guessed from extension if None)
        """
        if format is None:
            # Guess format from file extension
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            try:
                format = DataFormat(ext)
            except ValueError:
                raise ValueError(f"Unsupported file extension: {ext}")
        
        self.logger.info(f"Saving data to {path} as {format}")
        
        if format == DataFormat.CSV:
            df.to_csv(path, index=False)
        elif format == DataFormat.EXCEL:
            df.to_excel(path, index=False)
        elif format == DataFormat.JSON:
            df.to_json(path, orient='records')
        elif format == DataFormat.PARQUET:
            df.to_parquet(path, index=False)
        elif format == DataFormat.SQL:
            # This would require additional parameters in practice
            raise NotImplementedError("SQL saving not implemented yet")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a dataset and return summary statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Summary statistics
        """
        self.logger.info(f"Analyzing dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Basic info
        info = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
            "missing_values": {},
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }
        
        # Column info
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "missing_values": int(df[col].isna().sum()),
            }
            
            # Add descriptive statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                })
            
            info["columns"][col] = col_info
            
            # Track missing values
            missing = int(df[col].isna().sum())
            if missing > 0:
                info["missing_values"][col] = missing
        
        return info
    
    def detect_anomalies(self, df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """Detect anomalies in numerical columns.
        
        Args:
            df: DataFrame to analyze
            method: Anomaly detection method ('zscore', 'iqr')
            threshold: Detection threshold
            
        Returns:
            DataFrame with anomaly flags
        """
        self.logger.info(f"Detecting anomalies using {method} method")
        
        # Start with a copy of the dataframe
        result = df.copy()
        
        # Only process numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if method == "zscore":
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    continue
                z_scores = (df[col] - mean) / std
                result[f"{col}_anomaly"] = abs(z_scores) > threshold
            
            elif method == "iqr":
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                result[f"{col}_anomaly"] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            else:
                raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        return result
    
    def fill_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """Fill missing values in the dataset.
        
        Args:
            df: DataFrame to process
            strategy: Dict mapping column names to fill strategies
                      ('mean', 'median', 'mode', 'constant:value')
                      
        Returns:
            DataFrame with filled values
        """
        self.logger.info("Filling missing values")
        
        if strategy is None:
            strategy = {}
        
        # Start with a copy of the dataframe
        result = df.copy()
        
        for col in df.columns:
            if col in strategy:
                fill_method = strategy[col]
                
                if fill_method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    result[col] = df[col].fillna(df[col].mean())
                
                elif fill_method == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    result[col] = df[col].fillna(df[col].median())
                
                elif fill_method == "mode":
                    result[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
                
                elif fill_method.startswith("constant:"):
                    value = fill_method.split(":", 1)[1]
                    # Convert to appropriate type if possible
                    if pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            value = float(value)
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            pass
                    result[col] = df[col].fillna(value)
                
                else:
                    self.logger.warning(f"Unknown fill strategy for column {col}: {fill_method}")
        
        return result

class DatasetSplitter:
    """Split datasets for machine learning."""
    
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.0):
        """Initialize the dataset splitter.
        
        Args:
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
        """
        self.test_size = test_size
        self.validation_size = validation_size
    
    def train_test_split(self, df: pd.DataFrame, stratify_column: str = None) -> Dict[str, pd.DataFrame]:
        """Split data into training and test sets.
        
        Args:
            df: DataFrame to split
            stratify_column: Column to use for stratified sampling
            
        Returns:
            Dict with 'train' and 'test' dataframes
        """
        from sklearn.model_selection import train_test_split
        
        if stratify_column and stratify_column in df.columns:
            stratify = df[stratify_column]
        else:
            stratify = None
        
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            stratify=stratify,
            random_state=42
        )
        
        return {
            "train": train_df,
            "test": test_df
        }
    
    def train_validation_test_split(self, df: pd.DataFrame, stratify_column: str = None) -> Dict[str, pd.DataFrame]:
        """Split data into training, validation, and test sets.
        
        Args:
            df: DataFrame to split
            stratify_column: Column to use for stratified sampling
            
        Returns:
            Dict with 'train', 'validation', and 'test' dataframes
        """
        from sklearn.model_selection import train_test_split
        
        if stratify_column and stratify_column in df.columns:
            stratify = df[stratify_column]
        else:
            stratify = None
        
        # First split: train+validation vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            stratify=stratify,
            random_state=42
        )
        
        # Update stratify for next split
        if stratify_column and stratify_column in df.columns:
            stratify = train_val_df[stratify_column]
        else:
            stratify = None
        
        # Second split: train vs validation
        # Calculate effective validation size
        validation_size_adjusted = self.validation_size / (1 - self.test_size)
        
        # Perform split
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=validation_size_adjusted, 
            stratify=stratify,
            random_state=42
        )
        
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }

class FeatureEngineering:
    """Feature engineering tools for data preprocessing."""
    
    def create_datetime_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Extract datetime features from a date column.
        
        Args:
            df: DataFrame to process
            date_column: Name of the date column
            
        Returns:
            DataFrame with new features
        """
        # Make sure the column is datetime
        if date_column not in df.columns:
            raise ValueError(f"Column not found: {date_column}")
        
        result = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(result[date_column]):
            result[date_column] = pd.to_datetime(result[date_column], errors='coerce')
        
        # Extract features
        result[f"{date_column}_year"] = result[date_column].dt.year
        result[f"{date_column}_month"] = result[date_column].dt.month
        result[f"{date_column}_day"] = result[date_column].dt.day
        result[f"{date_column}_weekday"] = result[date_column].dt.weekday
        result[f"{date_column}_quarter"] = result[date_column].dt.quarter
        result[f"{date_column}_is_month_end"] = result[date_column].dt.is_month_end
        result[f"{date_column}_is_month_start"] = result[date_column].dt.is_month_start
        
        return result
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = "onehot") -> pd.DataFrame:
        """Encode categorical variables.
        
        Args:
            df: DataFrame to process
            columns: List of categorical columns to encode
            method: Encoding method ('onehot', 'label', 'target')
            
        Returns:
            DataFrame with encoded variables
        """
        result = df.copy()
        
        if method == "onehot":
            # One-hot encoding
            for col in columns:
                if col in result.columns:
                    dummies = pd.get_dummies(result[col], prefix=col, drop_first=False)
                    result = pd.concat([result, dummies], axis=1)
                    result.drop(col, axis=1, inplace=True)
        
        elif method == "label":
            # Label encoding
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                if col in result.columns:
                    le = LabelEncoder()
                    result[col] = le.fit_transform(result[col].astype(str))
        
        elif method == "target":
            # Target encoding would require a target variable
            raise NotImplementedError("Target encoding not implemented yet")
        
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
        
        return result
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str] = None, method: str = "minmax") -> pd.DataFrame:
        """Normalize numerical features.
        
        Args:
            df: DataFrame to process
            columns: List of columns to normalize (or all numeric if None)
            method: Normalization method ('minmax', 'standard', 'robust')
            
        Returns:
            DataFrame with normalized features
        """
        result = df.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = result.select_dtypes(include=['number']).columns.tolist()
        
        if method == "minmax":
            # Min-max scaling
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            result[columns] = scaler.fit_transform(result[columns])
        
        elif method == "standard":
            # Standardization (z-score)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            result[columns] = scaler.fit_transform(result[columns])
        
        elif method == "robust":
            # Robust scaling (using median and IQR)
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            result[columns] = scaler.fit_transform(result[columns])
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        return result
"""

def main():
    """Run the semantic search demo."""
    # Create a project directory
    project_dir = Path("semantic_search_demo_project")
    project_dir.mkdir(exist_ok=True)
    
    # Configure LlamaIndex settings
    print("Configuring LlamaIndex settings...")
    config.set("llama_index", "enabled", True)
    config.set("llama_index", "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a knowledge graph with LlamaIndex integration
    print("Creating LlamaIndex-powered knowledge graph...")
    graph = KnowledgeGraph("semantic_demo", storage_dir=str(project_dir))
    
    # Create a code analyzer
    analyzer = CodeAnalyzer(graph)
    
    # Analyze the data processing code
    print("\nAnalyzing data processing code...")
    component_id = analyzer.analyze_code(
        code_text=DATA_PROCESSING_CODE,
        file_path="data_processing.py"
    )
    
    # Print information about components
    print("\nComponents in the knowledge graph:")
    components = {}
    for node_id, data in graph.graph.nodes(data=True):
        name = data.get("name", "Unknown")
        comp_type = data.get("type", "Unknown")
        components[node_id] = (name, comp_type)
        print(f"- [{comp_type}] {name}")
    
    # Create context assembler
    context_assembler = ContextAssembler(graph)
    
    # Demo semantic searches
    print("\n=== Semantic Search Demo ===")
    
    # Example 1: Search for code related to anomaly detection
    search_queries = [
        "anomaly detection in data",
        "loading data from files",
        "fill missing values in dataset",
        "process categorical variables",
        "normalize numeric columns",
        "split data for machine learning",
        "extract features from dates"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        search_results = graph.search_code(
            query=query,
            limit=2
        )
        
        if search_results:
            print("Results:")
            for i, result in enumerate(search_results):
                print(f"{i+1}. [{result['type']}] {result['name']}: {result['description']}")
        else:
            print("No results found.")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
