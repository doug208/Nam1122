
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


# Test import of the realesrgan_upscaler module
import sys
sys.path.insert(0, ".")

try:
    from services.upscaling.realesrgan_upscaler import upscale_video_realesrgan
    print("✅ Successfully imported upscale_video_realesrgan")
    print(f"Function signature: {upscale_video_realesrgan.__code__.co_varnames[:4]}")
except ImportError as e:
    print(f"⚠️ Import error (expected if realesrgan not installed): {e}")
except Exception as e:
    print(f"❌ Error: {e}")