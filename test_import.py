
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
"""Test that the server module can be imported without errors."""

import sys
sys.path.insert(0, '.')

# Test basic import
try:
    from services.upscaling import server
    print("✓ Module imported successfully")
    
    # Check that _model_cache exists
    assert hasattr(server, '_model_cache'), "_model_cache not found"
    assert isinstance(server._model_cache, dict), "_model_cache is not a dict"
    print("✓ _model_cache exists and is a dict")
    
    # Check that get_or_load_model exists
    assert hasattr(server, 'get_or_load_model'), "get_or_load_model not found"
    print("✓ get_or_load_model function exists")
    
    # Check that startup_event exists
    assert hasattr(server, 'startup_event'), "startup_event not found"
    print("✓ startup_event function exists")
    
    # Check that app exists
    assert hasattr(server, 'app'), "app not found"
    print("✓ FastAPI app exists")
    
    print("\n✅ All checks passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)