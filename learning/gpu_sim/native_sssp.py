"""Small owning binding to the public libcugraph C traversal API.

cuGraph 26.06's Python vertex-check wrappers omit native array frees. The
high-level SSSP check leaks a full vertex list on every call; the lower-level
Python SSSP check also leaks its membership result. Call the same C algorithm
directly, with explicit ownership of handles, graphs, results, and array views.
No dependency patch, private Python object layout, or runtime compilation is
needed. Declarations match the headers shipped with libcugraph-cu13 26.06.
"""

import ctypes as ct
from functools import lru_cache
import weakref


class _GraphProperties(ct.Structure):
    # bool_t in cugraph_c/types.h is a C enum, not a one-byte C++ bool.
    _fields_ = [("is_symmetric", ct.c_int), ("is_multigraph", ct.c_int)]


@lru_cache(maxsize=1)
def _library():
    import libcugraph

    libraries = libcugraph.load_library()
    lib = next((item for item in libraries
                if hasattr(item, "cugraph_sssp")), None)
    if lib is None:
        raise RuntimeError("the installed libcugraph C API could not be loaded")
    ptr, out = ct.c_void_p, ct.POINTER(ct.c_void_p)
    declarations = {
        "cugraph_create_resource_handle": (ptr, [ptr]),
        "cugraph_free_resource_handle": (None, [ptr]),
        "cugraph_graph_create_sg": (ct.c_int, [
            ptr, ct.POINTER(_GraphProperties), ptr, ptr, ptr, ptr, ptr, ptr,
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, out, out]),
        "cugraph_graph_free": (None, [ptr]),
        "cugraph_sssp": (ct.c_int, [
            ptr, ptr, ct.c_size_t, ct.c_double, ct.c_int, ct.c_int, out, out]),
        "cugraph_paths_result_free": (None, [ptr]),
        "cugraph_type_erased_device_array_view_create": (
            ptr, [ptr, ct.c_size_t, ct.c_int]),
        "cugraph_type_erased_device_array_view_free": (None, [ptr]),
        "cugraph_type_erased_device_array_view_size": (ct.c_size_t, [ptr]),
        "cugraph_type_erased_device_array_view_type": (ct.c_int, [ptr]),
        "cugraph_type_erased_device_array_view_copy": (
            ct.c_int, [ptr, ptr, ptr, out]),
        "cugraph_error_message": (ct.c_char_p, [ptr]),
        "cugraph_error_free": (None, [ptr]),
    }
    for name in ("vertices", "distances", "predecessors"):
        declarations[f"cugraph_paths_result_get_{name}"] = (ptr, [ptr])
    for name, (result, arguments) in declarations.items():
        function = getattr(lib, name)
        function.restype = result
        function.argtypes = arguments
    return lib


def _check(lib, status, error):
    if not status:
        return
    try:
        raw = lib.cugraph_error_message(error) if error else None
        message = raw.decode("utf-8", "replace") if raw else f"libcugraph error {status}"
    finally:
        if error:
            lib.cugraph_error_free(error)
    if status == 3:  # CUGRAPH_ALLOC_ERROR
        raise MemoryError(message)
    raise RuntimeError(message)


class NativeSSSP:
    """One reusable native resource handle, shared by its owned graphs."""

    def __init__(self):
        self.lib = _library()
        self.handle = self.lib.cugraph_create_resource_handle(None)
        if not self.handle:
            raise RuntimeError("libcugraph could not create a resource handle")
        self._finalize = weakref.finalize(
            self, self.lib.cugraph_free_resource_handle, self.handle)

    def graph(self, num_nodes, sources, destinations, weights):
        return _NativeGraph(self, num_nodes, sources, destinations, weights)

    def _view(self, array):
        # The router supplies contiguous int32 IDs and float32/float64 costs.
        types = {"int32": 2, "int64": 3, "float32": 8, "float64": 9}
        if array.ndim != 1 or not array.flags.c_contiguous:
            raise ValueError("native graph arrays must be contiguous vectors")
        view = self.lib.cugraph_type_erased_device_array_view_create(
            array.data.ptr, array.size, types[str(array.dtype)])
        if not view:
            raise MemoryError("libcugraph could not create an array view")
        return view

    def _copy_result(self, view):
        import cupy as cp

        if not view:
            raise RuntimeError("libcugraph returned a null SSSP result view")
        lib = self.lib
        destination = None
        try:
            dtype = {2: cp.int32, 3: cp.int64, 8: cp.float32, 9: cp.float64}[
                lib.cugraph_type_erased_device_array_view_type(view)]
            array = cp.empty(
                lib.cugraph_type_erased_device_array_view_size(view), dtype=dtype)
            destination = self._view(array)
            error = ct.c_void_p()
            _check(lib, lib.cugraph_type_erased_device_array_view_copy(
                self.handle, destination, view, ct.byref(error)), error)
            return array
        finally:
            if destination:
                lib.cugraph_type_erased_device_array_view_free(destination)
            lib.cugraph_type_erased_device_array_view_free(view)


class _NativeGraph:
    def __init__(self, context, num_nodes, sources, destinations, weights):
        import cupy as cp

        self.context = context  # The handle must outlive this graph.
        self.num_nodes = int(num_nodes)
        if not 0 < self.num_nodes < 2 ** 31:
            raise ValueError("native terrain graphs require positive int32 node IDs")
        if not (sources.size == destinations.size == weights.size):
            raise ValueError("native graph edge arrays must have equal lengths")
        lib = context.lib
        views = []
        graph, error = ct.c_void_p(), ct.c_void_p()
        try:
            # Include isolated/blocked nodes so the source range check below
            # proves membership without extracting a full native vertex list.
            vertices = cp.arange(self.num_nodes, dtype=cp.int32)
            for array in (vertices, sources, destinations, weights):
                views.append(context._view(array))
            cp.cuda.get_current_stream().synchronize()
            properties = _GraphProperties(0, 0)
            _check(lib, lib.cugraph_graph_create_sg(
                context.handle, ct.byref(properties), *views, None, None,
                0, 0, 0, 0, 0, 0, ct.byref(graph), ct.byref(error)), error)
        except BaseException:
            if graph:
                lib.cugraph_graph_free(graph)
            raise
        finally:
            for view in views:
                lib.cugraph_type_erased_device_array_view_free(view)
        self.pointer = graph
        self._finalize = weakref.finalize(self, lib.cugraph_graph_free, graph)

    def sssp(self, source):
        if not self._finalize.alive:
            raise RuntimeError("native graph has already been closed")
        if not 0 <= int(source) < self.num_nodes:
            raise ValueError("source is outside the native graph")
        lib = self.context.lib
        result, error = ct.c_void_p(), ct.c_void_p()
        try:
            _check(lib, lib.cugraph_sssp(
                self.context.handle, self.pointer, int(source), float("inf"),
                1, 0, ct.byref(result), ct.byref(error)), error)
            return tuple(self.context._copy_result(
                getattr(lib, f"cugraph_paths_result_get_{name}")(result))
                for name in ("vertices", "distances", "predecessors"))
        finally:
            if result:
                lib.cugraph_paths_result_free(result)

    def close(self):
        self._finalize()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
