
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


import ast
import sys

# Check syntax of the new file
with open('services/upscaling/realesrgan_upscaler.py', 'r') as f:
    code = f.read()

try:
    ast.parse(code)
    print("✅ Syntax check passed for realesrgan_upscaler.py")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")

# Check server.py syntax
with open('services/upscaling/server.py', 'r') as f:
    code = f.read()

try:
    ast.parse(code)
    print("✅ Syntax check passed for server.py")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")