#!/usr/bin/env python3
"""
Visual-to-Code Translator for Agent Lightning
Converts visual block programs to executable code
"""

import os
import sys
import json
import ast
import textwrap
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualBlock,
    BlockType,
    BlockPort,
    VisualProgram,
    ConnectionType
)
from visual_code_blocks import InteractiveBlock


class TargetLanguage(Enum):
    """Supported target languages for code generation"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"


@dataclass
class CodeFragment:
    """A fragment of generated code"""
    code: str
    language: TargetLanguage
    indentation: int = 0
    requires_imports: List[str] = field(default_factory=list)
    requires_variables: List[str] = field(default_factory=list)
    
    def get_indented_code(self) -> str:
        """Get code with proper indentation"""
        indent_str = "    " * self.indentation
        lines = self.code.split('\n')
        return '\n'.join(indent_str + line if line.strip() else line 
                        for line in lines)


class BlockTranslator(ABC):
    """Abstract base class for block translators"""
    
    @abstractmethod
    def translate(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate a block to code"""
        pass
    
    @abstractmethod
    def supported_languages(self) -> List[TargetLanguage]:
        """Get supported languages for this translator"""
        pass


class PythonTranslator(BlockTranslator):
    """Translates blocks to Python code"""
    
    def supported_languages(self) -> List[TargetLanguage]:
        return [TargetLanguage.PYTHON]
    
    def translate(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate block to Python code"""
        
        # Map block types to translation methods
        translators = {
            BlockType.IF_CONDITION: self._translate_if,
            BlockType.FOR_LOOP: self._translate_for_loop,
            BlockType.WHILE_LOOP: self._translate_while_loop,
            BlockType.FUNCTION_DEF: self._translate_function,
            BlockType.VARIABLE: self._translate_variable,
            BlockType.ASSIGNMENT: self._translate_assignment,
            BlockType.EXPRESSION: self._translate_expression,
            BlockType.RETURN: self._translate_return,
            BlockType.OUTPUT: self._translate_output,
            BlockType.INPUT: self._translate_input,
            BlockType.TRY_CATCH: self._translate_try_catch,
            BlockType.API_CALL: self._translate_api_call,
            BlockType.FILE_READ: self._translate_file_read,
            BlockType.FILE_WRITE: self._translate_file_write,
            BlockType.COMMENT: self._translate_comment,
            BlockType.IMPORT: self._translate_import,
            BlockType.CLASS_DEF: self._translate_class,
        }
        
        translator = translators.get(block.block_type, self._translate_generic)
        return translator(block, context)
    
    def _translate_if(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate IF condition block"""
        condition = block.properties.get("condition_expression", "True")
        
        code = f"if {condition}:"
        
        # Add placeholders for then/else branches
        if context.get("include_body", True):
            code += "\n    # Then branch\n    pass"
            code += "\nelse:\n    # Else branch\n    pass"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_for_loop(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate FOR loop block"""
        var_name = block.properties.get("variable_name", "item")
        items_expr = self._get_input_value(block, "items", context) or "[]"
        
        code = f"for {var_name} in {items_expr}:"
        
        if context.get("include_body", True):
            code += "\n    # Loop body\n    pass"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_while_loop(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate WHILE loop block"""
        condition = block.properties.get("condition_expression", "True")
        
        code = f"while {condition}:"
        
        if context.get("include_body", True):
            code += "\n    # Loop body\n    pass"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_function(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate function definition block"""
        func_name = block.properties.get("function_name", "function")
        params = block.properties.get("parameters", [])
        return_type = block.properties.get("return_type", None)
        
        # Build parameter string
        param_str = ", ".join(params) if isinstance(params, list) else str(params)
        
        # Add type hints if available
        if return_type and return_type != "any":
            code = f"def {func_name}({param_str}) -> {return_type}:"
        else:
            code = f"def {func_name}({param_str}):"
        
        if context.get("include_body", True):
            code += '\n    """Function docstring"""\n    pass'
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_variable(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate variable block"""
        var_name = block.properties.get("variable_name", "var")
        var_type = block.properties.get("variable_type", "any")
        initial_value = block.properties.get("initial_value")
        
        # Get value from input port or use initial value
        value = self._get_input_value(block, "value", context) or initial_value
        
        if value is not None:
            code = f"{var_name} = {self._format_value(value)}"
        else:
            code = f"{var_name} = None"
        
        # Add type hint as comment
        if var_type != "any":
            code = f"{var_name}: {self._map_type(var_type)} = {self._format_value(value)}"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0),
            requires_variables=[var_name]
        )
    
    def _translate_assignment(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate assignment block"""
        target = block.properties.get("target", "var")
        value = self._get_input_value(block, "value", context) or "None"
        
        code = f"{target} = {value}"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_expression(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate expression block"""
        expr = block.properties.get("expression", "")
        
        return CodeFragment(
            code=expr,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_return(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate return statement"""
        value = block.properties.get("value") or self._get_input_value(block, "value", context)
        
        if value:
            code = f"return {self._format_value(value)}"
        else:
            code = "return"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_output(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate output/print block"""
        value = self._get_input_value(block, "value", context) or '""'
        format_type = block.properties.get("format", "text")
        
        if format_type == "json":
            code = f"print(json.dumps({value}, indent=2))"
            requires_imports = ["import json"]
        else:
            code = f"print({value})"
            requires_imports = []
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0),
            requires_imports=requires_imports
        )
    
    def _translate_input(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate input block"""
        prompt = block.properties.get("prompt", "Enter value: ")
        var_name = block.properties.get("variable_name", "user_input")
        
        code = f'{var_name} = input("{prompt}")'
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0),
            requires_variables=[var_name]
        )
    
    def _translate_try_catch(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate try-catch block"""
        exception_var = block.properties.get("exception_var", "e")
        exception_type = block.properties.get("exception_type", "Exception")
        
        code = "try:"
        
        if context.get("include_body", True):
            code += "\n    # Try block\n    pass"
            code += f"\nexcept {exception_type} as {exception_var}:"
            code += f"\n    # Handle exception\n    print(f'Error: {{{exception_var}}}')"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_api_call(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate API call block"""
        url = block.properties.get("url", "")
        method = block.properties.get("method", "GET")
        timeout = block.properties.get("timeout", 30)
        
        if method == "GET":
            code = f'response = requests.get("{url}", timeout={timeout})'
        elif method == "POST":
            body = self._get_input_value(block, "body", context) or "{}"
            code = f'response = requests.post("{url}", json={body}, timeout={timeout})'
        else:
            code = f'response = requests.request("{method}", "{url}", timeout={timeout})'
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0),
            requires_imports=["import requests"]
        )
    
    def _translate_file_read(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate file read block"""
        path = block.properties.get("path") or self._get_input_value(block, "path", context)
        var_name = block.properties.get("variable_name", "content")
        
        code = f"with open({self._format_value(path)}, 'r') as f:\n    {var_name} = f.read()"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0),
            requires_variables=[var_name]
        )
    
    def _translate_file_write(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate file write block"""
        path = block.properties.get("path") or self._get_input_value(block, "path", context)
        content = self._get_input_value(block, "content", context) or '""'
        
        code = f"with open({self._format_value(path)}, 'w') as f:\n    f.write({content})"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_comment(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate comment block"""
        comment_text = block.properties.get("comment_text", "")
        
        # Handle multi-line comments
        if '\n' in comment_text:
            lines = comment_text.split('\n')
            code = '\n'.join(f"# {line}" for line in lines)
        else:
            code = f"# {comment_text}"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_import(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate import block"""
        module_name = block.properties.get("module_name", "")
        import_as = block.properties.get("import_as", "")
        from_module = block.properties.get("from_module", "")
        
        if from_module:
            code = f"from {from_module} import {module_name}"
        elif import_as:
            code = f"import {module_name} as {import_as}"
        else:
            code = f"import {module_name}"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=0  # Imports always at top level
        )
    
    def _translate_class(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate class definition block"""
        class_name = block.properties.get("class_name", "MyClass")
        base_classes = block.properties.get("base_classes", [])
        
        if base_classes:
            base_str = ", ".join(base_classes)
            code = f"class {class_name}({base_str}):"
        else:
            code = f"class {class_name}:"
        
        if context.get("include_body", True):
            code += '\n    """Class docstring"""\n    \n    def __init__(self):\n        pass'
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_generic(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Generic translation for unknown block types"""
        return CodeFragment(
            code=f"# TODO: Implement {block.block_type.value}",
            language=TargetLanguage.PYTHON,
            indentation=context.get("indentation", 0)
        )
    
    def _get_input_value(self, block: VisualBlock, port_name: str, context: Dict[str, Any]) -> Optional[str]:
        """Get value from input port"""
        # In a real implementation, this would trace connections
        # For now, return placeholder
        port = next((p for p in block.input_ports if p.name == port_name), None)
        if port and port.value:
            return self._format_value(port.value)
        return None
    
    def _format_value(self, value: Any) -> str:
        """Format a value for Python code"""
        if value is None:
            return "None"
        elif isinstance(value, str):
            # Check if it's already a variable name or expression
            if value.startswith("$") or any(c in value for c in "()[]{}+-*/=<>"):
                return value
            return repr(value)
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return f"[{', '.join(self._format_value(v) for v in value)}]"
        elif isinstance(value, dict):
            items = [f"{self._format_value(k)}: {self._format_value(v)}" 
                    for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return repr(value)
    
    def _map_type(self, type_name: str) -> str:
        """Map generic type to Python type hint"""
        type_map = {
            "any": "Any",
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List",
            "object": "Dict",
            "null": "None"
        }
        return type_map.get(type_name, "Any")


class JavaScriptTranslator(BlockTranslator):
    """Translates blocks to JavaScript code"""
    
    def supported_languages(self) -> List[TargetLanguage]:
        return [TargetLanguage.JAVASCRIPT, TargetLanguage.TYPESCRIPT]
    
    def translate(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate block to JavaScript code"""
        
        translators = {
            BlockType.IF_CONDITION: self._translate_if,
            BlockType.FOR_LOOP: self._translate_for_loop,
            BlockType.FUNCTION_DEF: self._translate_function,
            BlockType.VARIABLE: self._translate_variable,
            BlockType.OUTPUT: self._translate_output,
            BlockType.COMMENT: self._translate_comment,
        }
        
        translator = translators.get(block.block_type, self._translate_generic)
        return translator(block, context)
    
    def _translate_if(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate IF condition to JavaScript"""
        condition = block.properties.get("condition_expression", "true")
        
        code = f"if ({condition}) {{"
        
        if context.get("include_body", True):
            code += "\n  // Then branch\n}"
            code += " else {\n  // Else branch\n}"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.JAVASCRIPT,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_for_loop(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate FOR loop to JavaScript"""
        var_name = block.properties.get("variable_name", "item")
        items_expr = block.properties.get("items_expression", "[]")
        
        code = f"for (const {var_name} of {items_expr}) {{"
        
        if context.get("include_body", True):
            code += "\n  // Loop body\n}"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.JAVASCRIPT,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_function(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate function to JavaScript"""
        func_name = block.properties.get("function_name", "function")
        params = block.properties.get("parameters", [])
        
        param_str = ", ".join(params) if isinstance(params, list) else str(params)
        
        if context.get("language") == TargetLanguage.TYPESCRIPT:
            return_type = block.properties.get("return_type", "any")
            code = f"function {func_name}({param_str}): {return_type} {{"
        else:
            code = f"function {func_name}({param_str}) {{"
        
        if context.get("include_body", True):
            code += "\n  // Function body\n}"
        
        return CodeFragment(
            code=code,
            language=context.get("language", TargetLanguage.JAVASCRIPT),
            indentation=context.get("indentation", 0)
        )
    
    def _translate_variable(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate variable to JavaScript"""
        var_name = block.properties.get("variable_name", "var")
        value = block.properties.get("initial_value")
        is_const = block.properties.get("is_const", False)
        
        keyword = "const" if is_const else "let"
        
        if value is not None:
            code = f"{keyword} {var_name} = {self._format_value(value)};"
        else:
            code = f"{keyword} {var_name};"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.JAVASCRIPT,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_output(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate output to JavaScript"""
        value = block.properties.get("value", '""')
        
        code = f"console.log({value});"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.JAVASCRIPT,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_comment(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Translate comment to JavaScript"""
        comment_text = block.properties.get("comment_text", "")
        
        if '\n' in comment_text:
            lines = comment_text.split('\n')
            code = "/*\n" + '\n'.join(f" * {line}" for line in lines) + "\n */"
        else:
            code = f"// {comment_text}"
        
        return CodeFragment(
            code=code,
            language=TargetLanguage.JAVASCRIPT,
            indentation=context.get("indentation", 0)
        )
    
    def _translate_generic(self, block: VisualBlock, context: Dict[str, Any]) -> CodeFragment:
        """Generic translation for unknown block types"""
        return CodeFragment(
            code=f"// TODO: Implement {block.block_type.value}",
            language=TargetLanguage.JAVASCRIPT,
            indentation=context.get("indentation", 0)
        )
    
    def _format_value(self, value: Any) -> str:
        """Format a value for JavaScript code"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            return json.dumps(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return f"[{', '.join(self._format_value(v) for v in value)}]"
        elif isinstance(value, dict):
            return json.dumps(value)
        else:
            return json.dumps(value)


class VisualToCodeTranslator:
    """Main translator for converting visual programs to code"""
    
    def __init__(self):
        self.translators: Dict[TargetLanguage, BlockTranslator] = {
            TargetLanguage.PYTHON: PythonTranslator(),
            TargetLanguage.JAVASCRIPT: JavaScriptTranslator(),
            TargetLanguage.TYPESCRIPT: JavaScriptTranslator(),
        }
        self.imports: Set[str] = set()
        self.variables: Set[str] = set()
    
    def translate_program(
        self,
        program: VisualProgram,
        target_language: TargetLanguage = TargetLanguage.PYTHON
    ) -> str:
        """Translate a complete visual program to code"""
        
        if target_language not in self.translators:
            raise ValueError(f"Unsupported language: {target_language}")
        
        translator = self.translators[target_language]
        
        # Reset state
        self.imports.clear()
        self.variables.clear()
        
        # Get blocks in execution order
        ordered_blocks = program.get_execution_order()
        
        # Group blocks by scope
        scopes = self._analyze_scopes(ordered_blocks, program.connections)
        
        # Generate code fragments
        fragments = []
        context = {
            "language": target_language,
            "indentation": 0,
            "include_body": True,
            "program": program
        }
        
        for scope in scopes:
            scope_fragments = self._translate_scope(scope, translator, context)
            fragments.extend(scope_fragments)
        
        # Assemble final code
        code = self._assemble_code(fragments, target_language)
        
        return code
    
    def _analyze_scopes(
        self,
        blocks: List[VisualBlock],
        connections: List[Dict[str, Any]]
    ) -> List[List[VisualBlock]]:
        """Analyze and group blocks by scope"""
        # For now, simple linear grouping
        # In a real implementation, would handle nested scopes
        return [blocks]
    
    def _translate_scope(
        self,
        blocks: List[VisualBlock],
        translator: BlockTranslator,
        context: Dict[str, Any]
    ) -> List[CodeFragment]:
        """Translate a scope of blocks"""
        fragments = []
        
        for block in blocks:
            # Update context based on connections
            self._update_context_for_block(block, context)
            
            # Translate block
            fragment = translator.translate(block, context)
            
            # Track imports and variables
            self.imports.update(fragment.requires_imports)
            self.variables.update(fragment.requires_variables)
            
            fragments.append(fragment)
        
        return fragments
    
    def _update_context_for_block(self, block: VisualBlock, context: Dict[str, Any]):
        """Update context based on block connections"""
        # Check if block is inside a control structure
        program = context.get("program")
        if program:
            for conn in program.connections:
                if conn["to_block"] == block.block_id:
                    if conn["type"] == ConnectionType.LOOP_BODY.value:
                        context["indentation"] = context.get("indentation", 0) + 1
                    elif conn["type"] == ConnectionType.FUNCTION_BODY.value:
                        context["indentation"] = 1
    
    def _assemble_code(self, fragments: List[CodeFragment], language: TargetLanguage) -> str:
        """Assemble code fragments into final code"""
        lines = []
        
        # Add header
        lines.append(self._generate_header(language))
        
        # Add imports
        if self.imports:
            lines.extend(sorted(self.imports))
            lines.append("")
        
        # Add main code
        if language == TargetLanguage.PYTHON:
            # Add main function wrapper
            lines.append("def main():")
            for fragment in fragments:
                code = fragment.get_indented_code()
                # Indent for main function
                indented = "\n".join("    " + line if line.strip() else line 
                                   for line in code.split('\n'))
                lines.append(indented)
            lines.append("")
            lines.append("if __name__ == '__main__':")
            lines.append("    main()")
        else:
            # JavaScript/TypeScript
            for fragment in fragments:
                lines.append(fragment.get_indented_code())
        
        return "\n".join(lines)
    
    def _generate_header(self, language: TargetLanguage) -> str:
        """Generate file header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if language == TargetLanguage.PYTHON:
            return f'''#!/usr/bin/env python3
"""
Generated by Agent Lightning Visual Code Builder
Generated at: {timestamp}
"""
'''
        elif language in [TargetLanguage.JAVASCRIPT, TargetLanguage.TYPESCRIPT]:
            return f'''/**
 * Generated by Agent Lightning Visual Code Builder
 * Generated at: {timestamp}
 */
'''
        else:
            return f"// Generated at: {timestamp}\n"
    
    def validate_translation(self, code: str, language: TargetLanguage) -> Tuple[bool, List[str]]:
        """Validate generated code"""
        errors = []
        
        if language == TargetLanguage.PYTHON:
            try:
                ast.parse(code)
                return True, []
            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
                return False, errors
        
        # For other languages, basic validation
        if not code.strip():
            errors.append("Generated code is empty")
            return False, errors
        
        return True, []


# Test the translator
def test_visual_to_code_translator():
    """Test the visual-to-code translator"""
    print("\n" + "="*60)
    print("Visual-to-Code Translator Test")
    print("="*60)
    
    from visual_code_builder import VisualProgram, BlockFactory
    
    # Create a simple program
    program = VisualProgram(name="Test Program")
    factory = BlockFactory()
    
    # Add blocks
    func = factory.create_function_block()
    func.properties["function_name"] = "calculate_sum"
    func.properties["parameters"] = ["a", "b"]
    program.add_block(func)
    
    var = factory.create_variable_block()
    var.properties["variable_name"] = "result"
    var.properties["initial_value"] = 0
    program.add_block(var)
    
    output = factory.create_output_block()
    output.properties["value"] = "result"
    program.add_block(output)
    
    # Connect blocks
    program.connect_blocks(
        func.block_id, "body",
        var.block_id, "value",
        ConnectionType.FUNCTION_BODY
    )
    
    program.connect_blocks(
        var.block_id, "value",
        output.block_id, "value",
        ConnectionType.DATA_FLOW
    )
    
    print(f"\n📦 Created program with {len(program.blocks)} blocks")
    
    # Translate to Python
    translator = VisualToCodeTranslator()
    
    print("\n🐍 Translating to Python:")
    python_code = translator.translate_program(program, TargetLanguage.PYTHON)
    print(python_code)
    
    # Validate
    valid, errors = translator.validate_translation(python_code, TargetLanguage.PYTHON)
    print(f"\n✅ Python code valid: {valid}")
    if errors:
        for error in errors:
            print(f"   ❌ {error}")
    
    print("\n📜 Translating to JavaScript:")
    js_code = translator.translate_program(program, TargetLanguage.JAVASCRIPT)
    print(js_code)
    
    # Save generated code
    with open("generated_code.py", "w") as f:
        f.write(python_code)
    print(f"\n💾 Saved Python code to generated_code.py")
    
    return translator


if __name__ == "__main__":
    print("Visual-to-Code Translator for Agent Lightning")
    print("="*60)
    
    translator = test_visual_to_code_translator()
    
    print("\n✅ Visual-to-Code Translator ready!")