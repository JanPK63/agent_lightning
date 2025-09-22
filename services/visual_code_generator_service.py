#!/usr/bin/env python3
"""
Visual Code Generator Service
Standalone FastAPI service for generating code from visual project
specifications. Supports multiple programming languages: Python,
JavaScript, Java, Go
"""

import os
import sys
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sanitization utilities
from shared.sanitization import InputSanitizer, sanitize_user_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global sanitizer instance
sanitizer = InputSanitizer()


class ComponentSpec(BaseModel):
    """Specification for a visual component"""
    id: str = Field(description="Component unique identifier")
    type: str = Field(description="Component type (logic, data, ui, ai, integration, workflow)")
    name: Optional[str] = Field(default=None, description="Component name")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Component configuration")
    position: Optional[Dict[str, float]] = Field(default=None, description="Component position on canvas")


class ConnectionSpec(BaseModel):
    """Specification for a connection between components"""
    source_id: str = Field(description="Source component ID")
    source_port: str = Field(description="Source port name")
    target_id: str = Field(description="Target component ID")
    target_port: str = Field(description="Target port name")


class ProjectSpec(BaseModel):
    """Visual project specification for code generation"""
    name: str = Field(description="Project name")
    description: Optional[str] = Field(default="", description="Project description")
    components: List[ComponentSpec] = Field(default_factory=list, description="List of components")
    connections: List[ConnectionSpec] = Field(default_factory=list, description="List of connections")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    @validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate and sanitize project name"""
        if not v or not isinstance(v, str):
            raise ValueError("Project name must be a non-empty string")
        sanitized = sanitize_user_input(v, "text")
        if len(sanitized) > 100:
            raise ValueError("Project name must be 100 characters or less")
        return sanitized

    @validator('description')
    @classmethod
    def validate_description(cls, v):
        """Validate and sanitize project description"""
        if v is None:
            return ""
        sanitized = sanitize_user_input(v, "text")
        if len(sanitized) > 500:
            raise ValueError("Project description must be 500 characters or less")
        return sanitized


class CodeGenerationRequest(BaseModel):
    """Request for code generation"""
    project: ProjectSpec = Field(description="Visual project specification")
    language: str = Field(default="python", description="Target programming language")
    optimize: bool = Field(default=True, description="Whether to optimize generated code")
    include_comments: bool = Field(default=True, description="Include explanatory comments")
    package_name: Optional[str] = Field(default=None, description="Package/module name for generated code")

    @validator('language')
    @classmethod
    def validate_language(cls, v):
        """Validate target language"""
        supported_languages = ["python", "javascript", "java", "go"]
        if v.lower() not in supported_languages:
            langs = ", ".join(supported_languages)
            raise ValueError(f"Unsupported language: {v}. Supported: {langs}")
        return v.lower()

    @validator('package_name')
    @classmethod
    def validate_package_name(cls, v):
        """Validate and sanitize package name"""
        if v is None:
            return None
        sanitized = sanitize_user_input(v, "text")
        if len(sanitized) > 50:
            raise ValueError("Package name must be 50 characters or less")
        return sanitized


class CodeGenerationResponse(BaseModel):
    """Response containing generated code"""
    code_id: str = Field(description="Unique identifier for the generated code")
    language: str = Field(description="Programming language of generated code")
    code: str = Field(description="Generated code content")
    lines: int = Field(description="Number of lines in generated code")
    timestamp: str = Field(description="Generation timestamp")
    metadata: Dict[str, Any] = Field(description="Additional metadata about generation")


class HealthResponse(BaseModel):
    """Health check response"""
    service: str = Field(description="Service name")
    status: str = Field(description="Service status")
    version: str = Field(description="Service version")
    timestamp: str = Field(description="Current timestamp")
    supported_languages: List[str] = Field(description="Supported programming languages")


class CodeTranslator:
    """Translate visual projects to code in multiple programming languages"""

    def __init__(self):
        self.supported_languages = ["python", "javascript", "java", "go"]
        self.templates = self._load_language_templates()

    def _load_language_templates(self) -> Dict[str, Dict[str, str]]:
        """Load code templates for different languages"""
        return {
            "python": {
                "class_template": """class {class_name}:
    \"\"\"{description}\"\"\"

    def __init__(self):
        {init_body}

    {methods}
""",
                "method_template": """    def {method_name}(self{params}) -> {return_type}:
        \"\"\"{description}\"\"\"
        {method_body}
""",
                "import_template": "import {module}\n",
                "main_template": """if __name__ == "__main__":
    {main_body}
"""
            },
            "javascript": {
                "class_template": """class {class_name} {
    constructor() {
        {init_body}
    }

    {methods}
}
""",
                "method_template": """    {method_name}({params}) {
        {method_body}
    }
""",
                "import_template": "import {module} from '{source}';\n",
                "main_template": """// Main execution
{main_body}
"""
            },
            "java": {
                "class_template": """public class {class_name} {
    {fields}

    public {class_name}() {
        {init_body}
    }

    {methods}
}
""",
                "method_template": """    public {return_type} {method_name}({params}) {
        {method_body}
    }
""",
                "import_template": "import {module};\n",
                "main_template": """    public static void main(String[] args) {
        {main_body}
    }
"""
            },
            "go": {
                "class_template": """type {class_name} struct {
    {fields}
}

func New{class_name}() *{class_name} {
    return &{class_name}{
        {init_body}
    }
}

{methods}
""",
                "method_template": """func (c *{class_name}) {method_name}({params}) {return_type} {
    {method_body}
}
""",
                "import_template": "import \"{module}\"\n",
                "main_template": """func main() {
    {main_body}
}
"""
            }
        }

    def translate_project(self, project: ProjectSpec, language: str = "python",
                         optimize: bool = True, include_comments: bool = True,
                         package_name: Optional[str] = None) -> str:
        """Translate visual project to code"""
        if language not in self.supported_languages:
            language = "python"

        if language == "python":
            return self._generate_python_code(project, optimize, include_comments, package_name)
        elif language == "javascript":
            return self._generate_javascript_code(project, optimize, include_comments, package_name)
        elif language == "java":
            return self._generate_java_code(project, optimize, include_comments, package_name)
        elif language == "go":
            return self._generate_go_code(project, optimize, include_comments, package_name)
        else:
            return self._generate_basic_code(project, language, include_comments)

    def _generate_python_code(self, project: ProjectSpec, optimize: bool = True,
                             include_comments: bool = True, package_name: Optional[str] = None) -> str:
        """Generate Python code from visual project"""
        code_lines = [
            "#!/usr/bin/env python3",
            f'"""{project.name} - Generated from Visual Code Builder"""',
            "",
            "import asyncio",
            "import logging",
            "from typing import Any, Dict, List, Optional",
            "",
            "logger = logging.getLogger(__name__)",
            "",
        ]

        # Add package declaration if specified
        if package_name:
            code_lines.insert(2, f"# Package: {package_name}")

        # Generate class based on project
        class_name = self._sanitize_identifier(project.name.replace(' ', ''))

        # Collect component information
        data_components = [c for c in project.components if c.type == "data"]
        logic_components = [c for c in project.components if c.type == "logic"]
        ai_components = [c for c in project.components if c.type == "ai"]

        # Generate class
        init_body = []
        methods = []

        # Initialize components
        for comp in project.components:
            init_body.append(f"        self.{comp.id} = {{}}  # {comp.type} component")

        # Generate methods for different component types
        for comp in data_components:
            methods.extend([
                f"    def get_{comp.id}(self) -> Dict[str, Any]:",
                f'        """Get data from {comp.name or comp.id} component"""',
                f"        return self.{comp.id}",
                "",
                f"    def set_{comp.id}(self, data: Dict[str, Any]) -> None:",
                f'        """Set data for {comp.name or comp.id} component"""',
                f"        self.{comp.id} = data",
                ""
            ])

        for comp in logic_components:
            methods.extend([
                f"    def process_{comp.id}(self, input_data: Dict[str, Any]) -> Dict[str, Any]:",
                f'        """Process logic in {comp.name or comp.id} component"""',
                f"        # TODO: Implement {comp.name or comp.id} logic",
                "        return input_data"
                ""
            ])

        for comp in ai_components:
            methods.extend([
                f"    def predict_{comp.id}(self, input_data: Dict[str, Any]) -> Any:",
                f'        """AI prediction from {comp.name or comp.id} component"""',
                f"        # TODO: Implement {comp.name or comp.id} AI logic",
                "        return None"
                ""
            ])

        # Add main execution method
        methods.extend([
            "    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:",
            '        """Execute the complete workflow"""',
            "        result = {}",
            ""
        ])

        # Add workflow execution based on connections
        for conn in project.connections:
            methods.append(f"        # Process connection: {conn.source_id} -> {conn.target_id}")

        methods.extend([
            "        return result",
            ""
        ])

        # Build class template
        class_template = f"""
class {class_name}:
    \"\"\"{project.description or 'Generated class from visual project'}\"\"\"

    def __init__(self):
{"".join(f"        {line}" for line in init_body)}

{"".join(methods)}
"""

        code_lines.append(class_template)

        # Add main execution block
        code_lines.extend([
            "",
            "if __name__ == '__main__':",
            f"    {class_name.lower()} = {class_name}()",
            "    asyncio.run({}.execute({{}}))".format(class_name.lower())
        ])

        return "\n".join(code_lines)

    def _generate_javascript_code(self, project: ProjectSpec, optimize: bool = True,
                                 include_comments: bool = True, package_name: Optional[str] = None) -> str:
        """Generate JavaScript code from visual project"""
        class_name = self._sanitize_identifier(project.name.replace(' ', ''))

        code_lines = [
            f"// {project.name} - Generated from Visual Code Builder",
            "",
        ]

        if package_name:
            code_lines.append(f"// Package: {package_name}")

        # Generate class
        init_body = []
        methods = []

        for comp in project.components:
            init_body.append(f"        this.{comp.id} = {{}}; // {comp.type} component")

        # Generate methods
        for comp in project.components:
            if comp.type == "data":
                methods.extend([
                    f"    get{comp.id[0].upper() + comp.id[1:]}() {{",
                    f"        return this.{comp.id};",
                    "    }",
                    "",
                    f"    set{comp.id[0].upper() + comp.id[1:]}(data) {{",
                    f"        this.{comp.id} = data;",
                    "    }",
                    ""
                ])
            elif comp.type == "logic":
                methods.extend([
                    f"    process{comp.id[0].upper() + comp.id[1:]}(inputData) {{",
                    f"        // TODO: Implement {comp.name or comp.id} logic",
                    "        return inputData;",
                    "    }",
                    ""
                ])

        # Build class
        class_code = f"""
class {class_name} {{
    constructor() {{
{"".join(f"        {line}" for line in init_body)}
    }}

{"".join(methods)}
}}
"""

        code_lines.append(class_code)

        # Add main execution
        code_lines.extend([
            "",
            "// Main execution",
            f"const {class_name.lower()} = new {class_name}();",
            f"console.log('{project.name} initialized');"
        ])

        return "\n".join(code_lines)

    def _generate_java_code(self, project: ProjectSpec, optimize: bool = True,
                           include_comments: bool = True, package_name: Optional[str] = None) -> str:
        """Generate Java code from visual project"""
        class_name = self._sanitize_identifier(project.name.replace(' ', ''))

        code_lines = [
            f"// {project.name} - Generated from Visual Code Builder",
            ""
        ]

        if package_name:
            code_lines.insert(0, f"package {package_name};")
            code_lines.insert(1, "")

        # Generate class
        fields = []
        init_body = []
        methods = []

        for comp in project.components:
            fields.append(f"    private Map<String, Object> {comp.id}; // {comp.type} component")
            init_body.append(f"        this.{comp.id} = new HashMap<>();")

        # Generate methods
        for comp in project.components:
            if comp.type == "data":
                methods.extend([
                    f"    public Map<String, Object> get{comp.id[0].upper() + comp.id[1:]}() {{",
                    f"        return this.{comp.id};",
                    "    }",
                    "",
                    f"    public void set{comp.id[0].upper() + comp.id[1:]}(Map<String, Object> data) {{",
                    f"        this.{comp.id} = data;",
                    "    }",
                    ""
                ])

        # Build class
        class_code = f"""
public class {class_name} {{
{"".join(fields)}

    public {class_name}() {{
{"".join(f"        {line}" for line in init_body)}
    }}

{"".join(methods)}
}}
"""

        code_lines.append(class_code)

        return "\n".join(code_lines)

    def _generate_go_code(self, project: ProjectSpec, optimize: bool = True,
                         include_comments: bool = True, package_name: Optional[str] = None) -> str:
        """Generate Go code from visual project"""
        class_name = self._sanitize_identifier(project.name.replace(' ', ''))

        code_lines = [
            f"// {project.name} - Generated from Visual Code Builder",
            ""
        ]

        if package_name:
            code_lines.insert(0, f"package {package_name}")
            code_lines.insert(1, "")
        else:
            code_lines.insert(0, "package main")
            code_lines.insert(1, "")

        # Generate struct
        fields = []
        init_body = []
        methods = []

        for comp in project.components:
            fields.append(f"    {comp.id} map[string]interface{{}} // {comp.type} component")
            init_body.append(f"        {comp.id}: make(map[string]interface{{}}),")

        # Generate methods
        for comp in project.components:
            if comp.type == "data":
                methods.extend([
                    f"func (c *{class_name}) Get{comp.id}(key string) interface{{}} {{",
                    f"    return c.{comp.id}[key]",
                    "}",
                    "",
                    f"func (c *{class_name}) Set{comp.id}(key string, value interface{{}}) {{",
                    f"    c.{comp.id}[key] = value",
                    "}",
                    ""
                ])

        # Build struct and constructor
        struct_code = f"""
type {class_name} struct {{
{"".join(fields)}
}}

func New{class_name}() *{class_name} {{
    return &{class_name}{{
{"".join(f"        {line}" for line in init_body)}
    }}
}}

{"".join(methods)}
"""

        code_lines.append(struct_code)

        return "\n".join(code_lines)

    def _generate_basic_code(self, project: ProjectSpec, language: str,
                           include_comments: bool = True) -> str:
        """Generate basic code structure (fallback)"""
        code_lines = [
            f"// {project.name} - Generated from Visual Code Builder",
            f"// Language: {language}",
            "",
            f"// Components: {len(project.components)}",
            f"// Connections: {len(project.connections)}",
            "",
            "// TODO: Implement full code generation for " + language,
            ""
        ]

        for comp in project.components:
            code_lines.append(f"// Component: {comp.id} ({comp.type})")

        return "\n".join(code_lines)

    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize identifier to be valid in target language"""
        # Remove invalid characters and ensure starts with letter
        sanitized = ''.join(c for c in identifier if c.isalnum() or c == '_')
        if not sanitized:
            sanitized = "GeneratedClass"
        if not sanitized[0].isalpha():
            sanitized = "Class" + sanitized
        return sanitized


class VisualCodeGeneratorService:
    """Main Visual Code Generator Service"""

    def __init__(self):
        self.app = FastAPI(
            title="Visual Code Generator Service",
            description="Generate code from visual project specifications",
            version="1.0.0"
        )

        # Initialize components
        self.translator = CodeTranslator()

        # Setup middleware
        self._setup_middleware()

        # Setup routes
        self._setup_routes()

        logger.info("✅ Visual Code Generator Service initialized")

    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                service="visual-code-generator",
                status="healthy",
                version="1.0.0",
                timestamp=datetime.utcnow().isoformat(),
                supported_languages=self.translator.supported_languages
            )

        @self.app.post("/generate", response_model=CodeGenerationResponse)
        async def generate_code(request: CodeGenerationRequest):
            """Generate code from visual project specification"""
            try:
                logger.info(f"Generating {request.language} code for project: {request.project.name}")

                # Generate code
                generated_code = self.translator.translate_project(
                    project=request.project,
                    language=request.language,
                    optimize=request.optimize,
                    include_comments=request.include_comments,
                    package_name=request.package_name
                )

                # Create response
                code_id = str(uuid.uuid4())
                response = CodeGenerationResponse(
                    code_id=code_id,
                    language=request.language,
                    code=generated_code,
                    lines=len(generated_code.split('\n')),
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "project_name": request.project.name,
                        "component_count": len(request.project.components),
                        "connection_count": len(request.project.connections),
                        "optimize": request.optimize,
                        "include_comments": request.include_comments
                    }
                )

                logger.info(f"Successfully generated {response.lines} lines of {request.language} code")
                return response

            except ValueError as e:
                logger.warning(f"Validation error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                raise HTTPException(status_code=500, detail="Internal server error during code generation")

        @self.app.get("/languages")
        async def get_supported_languages():
            """Get list of supported programming languages"""
            return {
                "languages": self.translator.supported_languages,
                "default": "python"
            }

        @self.app.get("/")
        async def root():
            """Root endpoint with service information"""
            return {
                "service": "Visual Code Generator Service",
                "version": "1.0.0",
                "description": "Generate code from visual project specifications",
                "endpoints": {
                    "GET /health": "Health check",
                    "POST /generate": "Generate code",
                    "GET /languages": "Get supported languages"
                },
                "supported_languages": self.translator.supported_languages
            }


def main():
    """Main entry point"""
    import uvicorn

    service = VisualCodeGeneratorService()

    port = int(os.getenv("CODE_GENERATOR_PORT", 8007))
    logger.info(f"Starting Visual Code Generator Service on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()