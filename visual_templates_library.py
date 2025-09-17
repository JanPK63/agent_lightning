#!/usr/bin/env python3
"""
Visual Code Templates Library for Agent Lightning
Pre-built visual program templates for common development patterns
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualProgram,
    BlockFactory,
    BlockType,
    VisualBlock,
    ConnectionType
)
from visual_component_library import ComponentTemplate


class TemplateCategory(Enum):
    """Categories of visual templates"""
    WEB_API = "web_api"
    DATA_PROCESSING = "data_processing"
    MACHINE_LEARNING = "machine_learning"
    MICROSERVICES = "microservices"
    AUTHENTICATION = "authentication"
    DATABASE = "database"
    FILE_OPERATIONS = "file_operations"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class VisualTemplate:
    """A complete visual program template"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: TemplateCategory = TemplateCategory.WEB_API
    tags: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    estimated_time: str = "30 minutes"
    program: Optional[VisualProgram] = None
    preview_image: Optional[str] = None
    documentation: str = ""
    requirements: List[str] = field(default_factory=list)
    
    def create_instance(self) -> VisualProgram:
        """Create a new instance of this template"""
        if self.program:
            # Clone the program
            program_json = self.program.to_json()
            new_program = VisualProgram.from_json(program_json)
            new_program.program_id = str(uuid.uuid4())
            new_program.name = f"{self.name} (Copy)"
            return new_program
        return VisualProgram(name=self.name)


class VisualTemplatesLibrary:
    """Library of pre-built visual code templates"""
    
    def __init__(self):
        self.templates: Dict[str, VisualTemplate] = {}
        self.factory = BlockFactory()
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load all default templates"""
        # REST API Templates
        self.add_template(self._create_rest_api_template())
        self.add_template(self._create_crud_api_template())
        self.add_template(self._create_webhook_handler_template())
        
        # Data Processing Templates
        self.add_template(self._create_data_pipeline_template())
        self.add_template(self._create_etl_pipeline_template())
        self.add_template(self._create_batch_processor_template())
        
        # Authentication Templates
        self.add_template(self._create_jwt_auth_template())
        self.add_template(self._create_oauth_flow_template())
        
        # Database Templates
        self.add_template(self._create_database_migration_template())
        self.add_template(self._create_query_optimizer_template())
        
        # Machine Learning Templates
        self.add_template(self._create_ml_training_template())
        self.add_template(self._create_prediction_api_template())
        
        # Microservices Templates
        self.add_template(self._create_microservice_template())
        self.add_template(self._create_event_driven_service_template())
        
        # File Operations Templates
        self.add_template(self._create_file_processor_template())
        self.add_template(self._create_csv_importer_template())
        
        # Testing Templates
        self.add_template(self._create_unit_test_template())
        self.add_template(self._create_integration_test_template())
        
        # Monitoring Templates
        self.add_template(self._create_health_check_template())
        self.add_template(self._create_metrics_collector_template())
    
    def add_template(self, template: VisualTemplate):
        """Add a template to the library"""
        self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[VisualTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: TemplateCategory) -> List[VisualTemplate]:
        """Get all templates in a category"""
        return [t for t in self.templates.values() if t.category == category]
    
    def search_templates(self, query: str) -> List[VisualTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    # Template Creation Methods
    
    def _create_rest_api_template(self) -> VisualTemplate:
        """Create REST API endpoint template"""
        program = VisualProgram(name="REST API Endpoint")
        
        # Function block for endpoint handler
        func = self.factory.create_function_block()
        func.properties["function_name"] = "handle_request"
        func.properties["parameters"] = ["request", "response"]
        program.add_block(func)
        
        # Input validation
        if_block = self.factory.create_if_block()
        if_block.properties["condition_expression"] = "validate_input(request)"
        program.add_block(if_block)
        
        # Database query
        db_block = self.factory.create_database_query_block()
        db_block.properties["query_type"] = "SELECT"
        program.add_block(db_block)
        
        # Transform data
        expr = self.factory.create_expression_block()
        expr.properties["expression"] = "transform_response(data)"
        program.add_block(expr)
        
        # Return response
        return_block = self.factory.create_return_block()
        return_block.properties["value"] = "response"
        program.add_block(return_block)
        
        # Connect blocks
        program.connect_blocks(func.block_id, "body", if_block.block_id, "condition")
        program.connect_blocks(if_block.block_id, "true_branch", db_block.block_id, "query")
        program.connect_blocks(db_block.block_id, "result", expr.block_id, "input")
        program.connect_blocks(expr.block_id, "result", return_block.block_id, "value")
        
        return VisualTemplate(
            name="REST API Endpoint",
            description="Basic REST API endpoint with validation and database query",
            category=TemplateCategory.WEB_API,
            tags=["api", "rest", "endpoint", "http"],
            difficulty="beginner",
            estimated_time="15 minutes",
            program=program,
            documentation="Creates a REST API endpoint with input validation, database query, and response transformation.",
            requirements=["fastapi", "sqlalchemy"]
        )
    
    def _create_crud_api_template(self) -> VisualTemplate:
        """Create CRUD API template"""
        program = VisualProgram(name="CRUD API")
        
        # Main router function
        func = self.factory.create_function_block()
        func.properties["function_name"] = "crud_router"
        func.properties["parameters"] = ["method", "resource", "data"]
        program.add_block(func)
        
        # Method routing
        if_create = self.factory.create_if_block()
        if_create.properties["condition_expression"] = "method == 'POST'"
        program.add_block(if_create)
        
        if_read = self.factory.create_if_block()
        if_read.properties["condition_expression"] = "method == 'GET'"
        program.add_block(if_read)
        
        if_update = self.factory.create_if_block()
        if_update.properties["condition_expression"] = "method == 'PUT'"
        program.add_block(if_update)
        
        if_delete = self.factory.create_if_block()
        if_delete.properties["condition_expression"] = "method == 'DELETE'"
        program.add_block(if_delete)
        
        # Database operations for each method
        for operation in ["INSERT", "SELECT", "UPDATE", "DELETE"]:
            db = self.factory.create_database_query_block()
            db.properties["query_type"] = operation
            program.add_block(db)
        
        return VisualTemplate(
            name="CRUD API",
            description="Complete Create, Read, Update, Delete API",
            category=TemplateCategory.WEB_API,
            tags=["crud", "api", "rest", "database"],
            difficulty="intermediate",
            estimated_time="45 minutes",
            program=program,
            requirements=["fastapi", "sqlalchemy", "pydantic"]
        )
    
    def _create_webhook_handler_template(self) -> VisualTemplate:
        """Create webhook handler template"""
        program = VisualProgram(name="Webhook Handler")
        
        # Webhook receiver
        func = self.factory.create_function_block()
        func.properties["function_name"] = "handle_webhook"
        func.properties["parameters"] = ["payload", "headers"]
        program.add_block(func)
        
        # Verify signature
        expr = self.factory.create_expression_block()
        expr.properties["expression"] = "verify_signature(payload, headers['signature'])"
        program.add_block(expr)
        
        # Process webhook
        try_block = self.factory.create_try_catch_block()
        program.add_block(try_block)
        
        # Queue for async processing
        api = self.factory.create_api_call_block()
        api.properties["url"] = "queue_service"
        api.properties["method"] = "POST"
        program.add_block(api)
        
        return VisualTemplate(
            name="Webhook Handler",
            description="Secure webhook handler with signature verification",
            category=TemplateCategory.WEB_API,
            tags=["webhook", "async", "security", "queue"],
            difficulty="intermediate",
            program=program,
            requirements=["hmac", "asyncio", "redis"]
        )
    
    def _create_data_pipeline_template(self) -> VisualTemplate:
        """Create data processing pipeline template"""
        program = VisualProgram(name="Data Pipeline")
        
        # Main pipeline function
        func = self.factory.create_function_block()
        func.properties["function_name"] = "process_data_pipeline"
        func.properties["parameters"] = ["input_source"]
        program.add_block(func)
        
        # Read data
        read = self.factory.create_file_read_block()
        read.properties["encoding"] = "utf-8"
        program.add_block(read)
        
        # Parse data
        expr_parse = self.factory.create_expression_block()
        expr_parse.properties["expression"] = "parse_csv(content)"
        program.add_block(expr_parse)
        
        # Filter data
        filter_expr = self.factory.create_expression_block()
        filter_expr.properties["expression"] = "filter(lambda x: x['valid'], data)"
        program.add_block(filter_expr)
        
        # Transform data
        for_loop = self.factory.create_for_loop_block()
        for_loop.properties["variable_name"] = "record"
        program.add_block(for_loop)
        
        # Aggregate results
        aggregate = self.factory.create_expression_block()
        aggregate.properties["expression"] = "aggregate_results(transformed_data)"
        program.add_block(aggregate)
        
        # Save results
        db_save = self.factory.create_database_query_block()
        db_save.properties["query_type"] = "INSERT"
        program.add_block(db_save)
        
        return VisualTemplate(
            name="Data Processing Pipeline",
            description="ETL pipeline for data processing with filtering and aggregation",
            category=TemplateCategory.DATA_PROCESSING,
            tags=["etl", "pipeline", "data", "csv", "aggregation"],
            difficulty="intermediate",
            estimated_time="30 minutes",
            program=program,
            documentation="Complete data pipeline: Read → Parse → Filter → Transform → Aggregate → Save",
            requirements=["pandas", "numpy"]
        )
    
    def _create_etl_pipeline_template(self) -> VisualTemplate:
        """Create ETL pipeline template"""
        program = VisualProgram(name="ETL Pipeline")
        
        # Extract phase
        func = self.factory.create_function_block()
        func.properties["function_name"] = "etl_pipeline"
        program.add_block(func)
        
        # Multiple data sources
        for source in ["database", "api", "file"]:
            if source == "database":
                block = self.factory.create_database_query_block()
            elif source == "api":
                block = self.factory.create_api_call_block()
            else:
                block = self.factory.create_file_read_block()
            program.add_block(block)
        
        # Transform phase
        transform = self.factory.create_expression_block()
        transform.properties["expression"] = "merge_and_transform(sources)"
        program.add_block(transform)
        
        # Validation
        validate = self.factory.create_if_block()
        validate.properties["condition_expression"] = "validate_data(transformed)"
        program.add_block(validate)
        
        # Load phase
        load = self.factory.create_database_query_block()
        load.properties["query_type"] = "INSERT"
        program.add_block(load)
        
        return VisualTemplate(
            name="ETL Pipeline",
            description="Extract, Transform, Load pipeline with multiple sources",
            category=TemplateCategory.DATA_PROCESSING,
            tags=["etl", "data", "integration", "warehouse"],
            difficulty="advanced",
            estimated_time="60 minutes",
            program=program,
            requirements=["pandas", "sqlalchemy", "requests"]
        )
    
    def _create_batch_processor_template(self) -> VisualTemplate:
        """Create batch processor template"""
        program = VisualProgram(name="Batch Processor")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "process_batch"
        func.properties["parameters"] = ["batch_size", "input_queue"]
        program.add_block(func)
        
        # While loop for continuous processing
        while_loop = self.factory.create_while_loop_block()
        while_loop.properties["condition_expression"] = "not queue.empty()"
        program.add_block(while_loop)
        
        # Get batch
        get_batch = self.factory.create_expression_block()
        get_batch.properties["expression"] = "queue.get_batch(batch_size)"
        program.add_block(get_batch)
        
        # Process in parallel
        for_loop = self.factory.create_for_loop_block()
        for_loop.properties["variable_name"] = "item"
        program.add_block(for_loop)
        
        # Error handling
        try_catch = self.factory.create_try_catch_block()
        program.add_block(try_catch)
        
        return VisualTemplate(
            name="Batch Processor",
            description="Process data in batches with error handling",
            category=TemplateCategory.DATA_PROCESSING,
            tags=["batch", "queue", "parallel", "processing"],
            difficulty="intermediate",
            program=program,
            requirements=["multiprocessing", "queue"]
        )
    
    def _create_jwt_auth_template(self) -> VisualTemplate:
        """Create JWT authentication template"""
        program = VisualProgram(name="JWT Authentication")
        
        # Login function
        login = self.factory.create_function_block()
        login.properties["function_name"] = "authenticate"
        login.properties["parameters"] = ["username", "password"]
        program.add_block(login)
        
        # Validate credentials
        validate = self.factory.create_if_block()
        validate.properties["condition_expression"] = "verify_credentials(username, password)"
        program.add_block(validate)
        
        # Generate token
        token = self.factory.create_expression_block()
        token.properties["expression"] = "generate_jwt_token(user_id, expires_in=3600)"
        program.add_block(token)
        
        # Return token
        return_block = self.factory.create_return_block()
        return_block.properties["value"] = "{'access_token': token, 'token_type': 'bearer'}"
        program.add_block(return_block)
        
        return VisualTemplate(
            name="JWT Authentication",
            description="JWT token-based authentication system",
            category=TemplateCategory.AUTHENTICATION,
            tags=["jwt", "auth", "security", "token"],
            difficulty="intermediate",
            program=program,
            requirements=["pyjwt", "passlib", "bcrypt"]
        )
    
    def _create_oauth_flow_template(self) -> VisualTemplate:
        """Create OAuth2 flow template"""
        program = VisualProgram(name="OAuth2 Flow")
        
        # OAuth initiation
        func = self.factory.create_function_block()
        func.properties["function_name"] = "oauth_flow"
        func.properties["parameters"] = ["provider", "redirect_uri"]
        program.add_block(func)
        
        # Generate state
        state = self.factory.create_expression_block()
        state.properties["expression"] = "generate_secure_state()"
        program.add_block(state)
        
        # Redirect to provider
        redirect = self.factory.create_api_call_block()
        redirect.properties["url"] = "provider_auth_url"
        redirect.properties["method"] = "GET"
        program.add_block(redirect)
        
        # Handle callback
        callback = self.factory.create_function_block()
        callback.properties["function_name"] = "oauth_callback"
        callback.properties["parameters"] = ["code", "state"]
        program.add_block(callback)
        
        # Exchange code for token
        exchange = self.factory.create_api_call_block()
        exchange.properties["url"] = "token_endpoint"
        exchange.properties["method"] = "POST"
        program.add_block(exchange)
        
        return VisualTemplate(
            name="OAuth2 Flow",
            description="Complete OAuth2 authentication flow",
            category=TemplateCategory.AUTHENTICATION,
            tags=["oauth", "oauth2", "authentication", "sso"],
            difficulty="advanced",
            program=program,
            requirements=["authlib", "requests", "cryptography"]
        )
    
    def _create_database_migration_template(self) -> VisualTemplate:
        """Create database migration template"""
        program = VisualProgram(name="Database Migration")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "migrate_database"
        func.properties["parameters"] = ["version"]
        program.add_block(func)
        
        # Check current version
        check = self.factory.create_database_query_block()
        check.properties["query_type"] = "SELECT"
        program.add_block(check)
        
        # Apply migrations
        for_loop = self.factory.create_for_loop_block()
        for_loop.properties["variable_name"] = "migration"
        program.add_block(for_loop)
        
        # Execute migration
        execute = self.factory.create_database_query_block()
        execute.properties["query_type"] = "EXECUTE"
        program.add_block(execute)
        
        return VisualTemplate(
            name="Database Migration",
            description="Database schema migration system",
            category=TemplateCategory.DATABASE,
            tags=["database", "migration", "schema", "sql"],
            difficulty="intermediate",
            program=program,
            requirements=["alembic", "sqlalchemy"]
        )
    
    def _create_query_optimizer_template(self) -> VisualTemplate:
        """Create query optimizer template"""
        program = VisualProgram(name="Query Optimizer")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "optimize_query"
        func.properties["parameters"] = ["query", "explain_plan"]
        program.add_block(func)
        
        # Analyze query
        analyze = self.factory.create_expression_block()
        analyze.properties["expression"] = "analyze_query_plan(explain_plan)"
        program.add_block(analyze)
        
        # Optimize based on patterns
        optimize = self.factory.create_if_block()
        optimize.properties["condition_expression"] = "has_optimization_opportunity"
        program.add_block(optimize)
        
        return VisualTemplate(
            name="Query Optimizer",
            description="Database query optimization tool",
            category=TemplateCategory.DATABASE,
            tags=["database", "optimization", "performance", "sql"],
            difficulty="advanced",
            program=program,
            requirements=["sqlparse", "sqlalchemy"]
        )
    
    def _create_ml_training_template(self) -> VisualTemplate:
        """Create ML training template"""
        program = VisualProgram(name="ML Training Pipeline")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "train_model"
        func.properties["parameters"] = ["dataset", "model_type", "hyperparameters"]
        program.add_block(func)
        
        # Load and preprocess data
        load = self.factory.create_file_read_block()
        program.add_block(load)
        
        preprocess = self.factory.create_expression_block()
        preprocess.properties["expression"] = "preprocess_data(data)"
        program.add_block(preprocess)
        
        # Split data
        split = self.factory.create_expression_block()
        split.properties["expression"] = "train_test_split(X, y, test_size=0.2)"
        program.add_block(split)
        
        # Train model
        train = self.factory.create_expression_block()
        train.properties["expression"] = "model.fit(X_train, y_train)"
        program.add_block(train)
        
        # Evaluate
        evaluate = self.factory.create_expression_block()
        evaluate.properties["expression"] = "evaluate_model(model, X_test, y_test)"
        program.add_block(evaluate)
        
        return VisualTemplate(
            name="ML Training Pipeline",
            description="Complete machine learning training pipeline",
            category=TemplateCategory.MACHINE_LEARNING,
            tags=["ml", "ai", "training", "model"],
            difficulty="advanced",
            estimated_time="90 minutes",
            program=program,
            requirements=["scikit-learn", "pandas", "numpy"]
        )
    
    def _create_prediction_api_template(self) -> VisualTemplate:
        """Create ML prediction API template"""
        program = VisualProgram(name="Prediction API")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "predict"
        func.properties["parameters"] = ["input_data"]
        program.add_block(func)
        
        # Load model
        load_model = self.factory.create_expression_block()
        load_model.properties["expression"] = "load_model('model.pkl')"
        program.add_block(load_model)
        
        # Preprocess input
        preprocess = self.factory.create_expression_block()
        preprocess.properties["expression"] = "preprocess_input(input_data)"
        program.add_block(preprocess)
        
        # Make prediction
        predict = self.factory.create_expression_block()
        predict.properties["expression"] = "model.predict(processed_data)"
        program.add_block(predict)
        
        # Format response
        format_resp = self.factory.create_expression_block()
        format_resp.properties["expression"] = "format_prediction_response(prediction)"
        program.add_block(format_resp)
        
        return VisualTemplate(
            name="ML Prediction API",
            description="REST API for machine learning predictions",
            category=TemplateCategory.MACHINE_LEARNING,
            tags=["ml", "api", "prediction", "inference"],
            difficulty="intermediate",
            program=program,
            requirements=["fastapi", "scikit-learn", "joblib"]
        )
    
    def _create_microservice_template(self) -> VisualTemplate:
        """Create microservice template"""
        program = VisualProgram(name="Microservice")
        
        # Service initialization
        init = self.factory.create_function_block()
        init.properties["function_name"] = "initialize_service"
        init.properties["parameters"] = ["config"]
        program.add_block(init)
        
        # Health check endpoint
        health = self.factory.create_function_block()
        health.properties["function_name"] = "health_check"
        program.add_block(health)
        
        # Service discovery registration
        register = self.factory.create_api_call_block()
        register.properties["url"] = "service_registry"
        register.properties["method"] = "POST"
        program.add_block(register)
        
        # Message handler
        handler = self.factory.create_function_block()
        handler.properties["function_name"] = "handle_message"
        handler.properties["parameters"] = ["message"]
        program.add_block(handler)
        
        return VisualTemplate(
            name="Microservice Template",
            description="Basic microservice with health check and service discovery",
            category=TemplateCategory.MICROSERVICES,
            tags=["microservice", "distributed", "service-discovery"],
            difficulty="intermediate",
            program=program,
            requirements=["fastapi", "consul", "pika"]
        )
    
    def _create_event_driven_service_template(self) -> VisualTemplate:
        """Create event-driven service template"""
        program = VisualProgram(name="Event-Driven Service")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "event_handler"
        func.properties["parameters"] = ["event_type", "payload"]
        program.add_block(func)
        
        # Event routing
        router = self.factory.create_if_block()
        router.properties["condition_expression"] = "event_type in handlers"
        program.add_block(router)
        
        # Process event
        process = self.factory.create_expression_block()
        process.properties["expression"] = "handlers[event_type](payload)"
        program.add_block(process)
        
        # Publish result
        publish = self.factory.create_api_call_block()
        publish.properties["url"] = "event_bus"
        publish.properties["method"] = "POST"
        program.add_block(publish)
        
        return VisualTemplate(
            name="Event-Driven Service",
            description="Event-driven microservice with pub/sub",
            category=TemplateCategory.MICROSERVICES,
            tags=["event-driven", "pub-sub", "messaging", "async"],
            difficulty="advanced",
            program=program,
            requirements=["rabbitmq", "kafka", "redis"]
        )
    
    def _create_file_processor_template(self) -> VisualTemplate:
        """Create file processor template"""
        program = VisualProgram(name="File Processor")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "process_files"
        func.properties["parameters"] = ["directory", "pattern"]
        program.add_block(func)
        
        # List files
        list_files = self.factory.create_expression_block()
        list_files.properties["expression"] = "glob.glob(f'{directory}/{pattern}')"
        program.add_block(list_files)
        
        # Process each file
        for_loop = self.factory.create_for_loop_block()
        for_loop.properties["variable_name"] = "filepath"
        program.add_block(for_loop)
        
        # Read file
        read = self.factory.create_file_read_block()
        program.add_block(read)
        
        # Process content
        process = self.factory.create_expression_block()
        process.properties["expression"] = "process_file_content(content)"
        program.add_block(process)
        
        return VisualTemplate(
            name="File Processor",
            description="Batch file processing with pattern matching",
            category=TemplateCategory.FILE_OPERATIONS,
            tags=["file", "batch", "processing", "io"],
            difficulty="beginner",
            program=program,
            requirements=["glob", "pathlib"]
        )
    
    def _create_csv_importer_template(self) -> VisualTemplate:
        """Create CSV importer template"""
        program = VisualProgram(name="CSV Importer")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "import_csv"
        func.properties["parameters"] = ["csv_file", "table_name"]
        program.add_block(func)
        
        # Read CSV
        read = self.factory.create_file_read_block()
        program.add_block(read)
        
        # Parse CSV
        parse = self.factory.create_expression_block()
        parse.properties["expression"] = "csv.DictReader(content)"
        program.add_block(parse)
        
        # Validate data
        validate = self.factory.create_for_loop_block()
        validate.properties["variable_name"] = "row"
        program.add_block(validate)
        
        # Insert to database
        insert = self.factory.create_database_query_block()
        insert.properties["query_type"] = "INSERT"
        program.add_block(insert)
        
        return VisualTemplate(
            name="CSV Importer",
            description="Import CSV data into database with validation",
            category=TemplateCategory.FILE_OPERATIONS,
            tags=["csv", "import", "database", "etl"],
            difficulty="beginner",
            program=program,
            requirements=["csv", "pandas", "sqlalchemy"]
        )
    
    def _create_unit_test_template(self) -> VisualTemplate:
        """Create unit test template"""
        program = VisualProgram(name="Unit Test")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "test_function"
        program.add_block(func)
        
        # Arrange
        setup = self.factory.create_expression_block()
        setup.properties["expression"] = "setup_test_data()"
        program.add_block(setup)
        
        # Act
        execute = self.factory.create_expression_block()
        execute.properties["expression"] = "function_under_test(test_input)"
        program.add_block(execute)
        
        # Assert
        assert_block = self.factory.create_if_block()
        assert_block.properties["condition_expression"] = "result == expected"
        program.add_block(assert_block)
        
        return VisualTemplate(
            name="Unit Test Template",
            description="Unit test with arrange-act-assert pattern",
            category=TemplateCategory.TESTING,
            tags=["test", "unit", "testing", "qa"],
            difficulty="beginner",
            program=program,
            requirements=["pytest", "unittest"]
        )
    
    def _create_integration_test_template(self) -> VisualTemplate:
        """Create integration test template"""
        program = VisualProgram(name="Integration Test")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "test_integration"
        program.add_block(func)
        
        # Setup test environment
        setup = self.factory.create_expression_block()
        setup.properties["expression"] = "setup_test_environment()"
        program.add_block(setup)
        
        # Start services
        start = self.factory.create_api_call_block()
        start.properties["url"] = "test_service"
        start.properties["method"] = "POST"
        program.add_block(start)
        
        # Run test scenario
        scenario = self.factory.create_expression_block()
        scenario.properties["expression"] = "run_test_scenario()"
        program.add_block(scenario)
        
        # Cleanup
        cleanup = self.factory.create_expression_block()
        cleanup.properties["expression"] = "cleanup_test_environment()"
        program.add_block(cleanup)
        
        return VisualTemplate(
            name="Integration Test",
            description="Integration test with setup and teardown",
            category=TemplateCategory.TESTING,
            tags=["test", "integration", "e2e", "testing"],
            difficulty="intermediate",
            program=program,
            requirements=["pytest", "requests", "docker"]
        )
    
    def _create_health_check_template(self) -> VisualTemplate:
        """Create health check template"""
        program = VisualProgram(name="Health Check")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "health_check"
        program.add_block(func)
        
        # Check database
        db_check = self.factory.create_database_query_block()
        db_check.properties["query_type"] = "SELECT"
        program.add_block(db_check)
        
        # Check external services
        api_check = self.factory.create_api_call_block()
        api_check.properties["url"] = "dependency_service/health"
        api_check.properties["method"] = "GET"
        program.add_block(api_check)
        
        # Aggregate health status
        aggregate = self.factory.create_expression_block()
        aggregate.properties["expression"] = "aggregate_health_status(checks)"
        program.add_block(aggregate)
        
        return VisualTemplate(
            name="Health Check Endpoint",
            description="Service health check with dependency monitoring",
            category=TemplateCategory.MONITORING,
            tags=["health", "monitoring", "status", "ops"],
            difficulty="beginner",
            program=program,
            requirements=["fastapi"]
        )
    
    def _create_metrics_collector_template(self) -> VisualTemplate:
        """Create metrics collector template"""
        program = VisualProgram(name="Metrics Collector")
        
        func = self.factory.create_function_block()
        func.properties["function_name"] = "collect_metrics"
        program.add_block(func)
        
        # Collect system metrics
        system = self.factory.create_expression_block()
        system.properties["expression"] = "get_system_metrics()"
        program.add_block(system)
        
        # Collect application metrics
        app = self.factory.create_expression_block()
        app.properties["expression"] = "get_application_metrics()"
        program.add_block(app)
        
        # Format for time series
        format_metrics = self.factory.create_expression_block()
        format_metrics.properties["expression"] = "format_metrics_for_prometheus(metrics)"
        program.add_block(format_metrics)
        
        # Push to monitoring system
        push = self.factory.create_api_call_block()
        push.properties["url"] = "prometheus_pushgateway"
        push.properties["method"] = "POST"
        program.add_block(push)
        
        return VisualTemplate(
            name="Metrics Collector",
            description="Collect and export metrics to monitoring system",
            category=TemplateCategory.MONITORING,
            tags=["metrics", "monitoring", "prometheus", "observability"],
            difficulty="intermediate",
            program=program,
            requirements=["prometheus-client", "psutil"]
        )
    
    def export_template(self, template_id: str, filepath: str):
        """Export a template to file"""
        template = self.get_template(template_id)
        if template:
            with open(filepath, 'w') as f:
                json.dump({
                    "template_id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category.value,
                    "tags": template.tags,
                    "difficulty": template.difficulty,
                    "program": template.program.to_json() if template.program else None,
                    "documentation": template.documentation,
                    "requirements": template.requirements
                }, f, indent=2)
    
    def import_template(self, filepath: str) -> Optional[VisualTemplate]:
        """Import a template from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            template = VisualTemplate(
                template_id=data.get("template_id", str(uuid.uuid4())),
                name=data["name"],
                description=data["description"],
                category=TemplateCategory(data["category"]),
                tags=data.get("tags", []),
                difficulty=data.get("difficulty", "intermediate"),
                documentation=data.get("documentation", ""),
                requirements=data.get("requirements", [])
            )
            
            if data.get("program"):
                template.program = VisualProgram.from_json(data["program"])
            
            self.add_template(template)
            return template
            
        except Exception as e:
            print(f"Error importing template: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics"""
        stats = {
            "total_templates": len(self.templates),
            "categories": {}
        }
        
        for category in TemplateCategory:
            templates = self.get_templates_by_category(category)
            stats["categories"][category.value] = len(templates)
        
        return stats


# Test the templates library
def test_templates_library():
    """Test the visual templates library"""
    print("\n" + "="*60)
    print("Visual Templates Library Test")
    print("="*60)
    
    library = VisualTemplatesLibrary()
    
    # Show statistics
    stats = library.get_statistics()
    print(f"\n📚 Template Library Statistics:")
    print(f"   Total Templates: {stats['total_templates']}")
    print(f"\n   By Category:")
    for category, count in stats['categories'].items():
        print(f"     • {category}: {count} templates")
    
    # Test search
    print(f"\n🔍 Searching for 'api' templates:")
    api_templates = library.search_templates("api")
    for template in api_templates[:5]:
        print(f"   • {template.name} ({template.category.value})")
        print(f"     {template.description}")
    
    # Test template instantiation
    print(f"\n🚀 Creating instance of REST API template:")
    rest_templates = library.get_templates_by_category(TemplateCategory.WEB_API)
    if rest_templates:
        template = rest_templates[0]
        instance = template.create_instance()
        print(f"   Created: {instance.name}")
        print(f"   Blocks: {len(instance.blocks)}")
        print(f"   Connections: {len(instance.connections)}")
    
    # Export a template
    if rest_templates:
        library.export_template(rest_templates[0].template_id, "rest_api_template.json")
        print(f"\n💾 Exported REST API template to rest_api_template.json")
    
    return library


if __name__ == "__main__":
    print("Visual Code Templates Library for Agent Lightning")
    print("="*60)
    
    library = test_templates_library()
    
    print("\n✅ Visual Templates Library ready!")
    print(f"\nAvailable templates help agents quickly create:")
    print("  • REST APIs with authentication")
    print("  • Data processing pipelines")
    print("  • Machine learning workflows")
    print("  • Microservices architectures")
    print("  • Database operations")
    print("  • Testing frameworks")
    print("  • And much more!")