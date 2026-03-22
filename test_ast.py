
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;


#!/usr/bin/env python3
"""Test that the server.py has correct AST structure."""

import ast
import sys

with open('services/upscaling/server.py', 'r') as f:
    source = f.read()

try:
    tree = ast.parse(source)
    print("✓ Python syntax is valid")
    
    # Find module-level assignments for _model_cache
    found_cache = False
    found_get_or_load_model = False
    found_startup_event = False
    
    for node in ast.walk(tree):
        # Check for _model_cache assignment
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '_model_cache':
                    found_cache = True
                    print("✓ Found _model_cache assignment at module level")
        
        # Check for get_or_load_model function
        if isinstance(node, ast.FunctionDef) and node.name == 'get_or_load_model':
            found_get_or_load_model = True
            print("✓ Found get_or_load_model function definition")
            # Check it has the right arguments
            args = node.args
            if len(args.args) == 2:
                print(f"  - Has 2 arguments: {args.args[0].arg}, {args.args[1].arg}")
        
        # Check for startup_event function
        if isinstance(node, ast.FunctionDef) and node.name == 'startup_event':
            found_startup_event = True
            print("✓ Found startup_event function definition")
            # Check it has the decorator
            if node.decorator_list:
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        if dec.func.attr == 'on_event':
                            print("  - Has @app.on_event decorator")
    
    # Check for get_or_load_model call in upscale_video
    found_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'get_or_load_model':
                found_call = True
                print("✓ Found get_or_load_model function call")
    
    if not found_cache:
        print("❌ _model_cache not found")
        sys.exit(1)
    if not found_get_or_load_model:
        print("❌ get_or_load_model function not found")
        sys.exit(1)
    if not found_startup_event:
        print("❌ startup_event function not found")
        sys.exit(1)
    if not found_call:
        print("❌ get_or_load_model call not found")
        sys.exit(1)
    
    print("\n✅ All AST checks passed!")
    
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)