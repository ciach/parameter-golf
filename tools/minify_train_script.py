#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if not body:
        return body
    first = body[0]
    if isinstance(first, ast.Expr):
        value = first.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
    return body


class DocstringStripper(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = _strip_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.body = _strip_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = _strip_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = _strip_docstring(node.body)
        return node


def minify_python(input_path: Path, output_path: Path) -> None:
    src = input_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(input_path))
    tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    minified = ast.unparse(tree).strip() + "\n"
    output_path.write_text(minified, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact trainer snapshot.")
    parser.add_argument("--input", required=True, help="Input python file.")
    parser.add_argument("--output", required=True, help="Output python file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    minify_python(input_path, output_path)
    compile(output_path.read_text(encoding="utf-8"), str(output_path), "exec")


if __name__ == "__main__":
    main()
