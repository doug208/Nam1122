
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


import sys
sys.path.insert(0, '.')

# Test importing the server module
try:
    import services.upscaling.server as server_module
    print("✅ Server module imports successfully")
    print(f"CUDA available: {server_module.torch.cuda.is_available()}")
    print(f"GPU config device: {server_module.GPU_CONFIG['device']}")
    print(f"Available models: {list(server_module.MODEL_CONFIGS.keys())}")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()