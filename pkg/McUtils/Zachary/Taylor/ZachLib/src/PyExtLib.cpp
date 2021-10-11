/*
    Implements some common useful utilities in writing Python extensions
*/

#include "PyExtLib.h"

/******************************** ARRAY MANIPULATION HELPERS ********************************/

Py_buffer _GetDataBuffer(PyObject *data) {

    Py_buffer view;
    PyObject_GetBuffer(data, &view, PyBUF_CONTIG_RO);
    return view;
}

void _CopyDataBuffer(PyObject *data, void *buff, long len, int dsize) {

    Py_buffer view_obj = _GetDataBuffer(data);
    Py_buffer *view = &view_obj;
    if (len) memcpy(view->buf, buff, len*dsize);

}

/****************************** STRING HANDLING **********************************/
#if PY_MAJOR_VERSION == 3
const char *_GetString( PyObject* s, const char *enc, const char *err, PyObject *pyStr) {
    pyStr = PyUnicode_AsEncodedString(s, enc, err);
    if (pyStr == NULL) return NULL;
    const char *strExcType =  PyBytes_AsString(pyStr);
//    Py_XDECREF(pyStr);
    return strExcType;
}
const char *_GetString( PyObject* s, PyObject *pyStr) {
    // unfortunately we need to pass the second pyStr so we can XDECREF it later
    return _GetString( s, "utf-8", "strict", pyStr); // utf-8 is safe since it contains ASCII fully
    }

const char *_Repr(PyObject *o, PyObject *repr) {
    PyObject *tmp = PyObject_Repr(o);
    PyObject *enc = PyUnicode_AsEncodedString(tmp, "utf-8", "strict");
    if ( enc == NULL) {
        Py_XDECREF(tmp);
        return NULL;
    }
    const char *str =  PyBytes_AsString(enc);
    Py_XDECREF(enc);
    Py_XDECREF(tmp);

    return str;
}

#else
const char *_GetString( PyObject* s ) {
    return PyString_AsString(s);
}
const char *_GetString( PyObject* s, PyObject *pyStr ) {
    // just to unify the 2/3 interface
    return _GetString( s );
    }
const char *_Repr(PyObject *o) {
    PyObject *tmp = PyObject_Repr(o);
    const char *str = _GetString(o);
    Py_XDECREF(tmp);
    return str;
}
const char *_Repr(PyObject *o, PyObject *repr) {
    return _Repr( o );
}
#endif

int _DebugPrint(int level, const char *fmt, ...) {
    if (level <= _DEBUG_LEVEL) {
        va_list args;
        va_start(args, fmt);

        int res = vprintf(fmt, args);
        printf("\n");

        fflush(stdout);

        return res;
    } else {
        return 0;
    }
}
int _DebugMessage(int level, const char *msg) {
    return _DebugPrint(level, "%s", msg);
}
int _DebugPrintObject(int lvl, PyObject *o){
    PyObject *repr = NULL;
    const char * buff=_Repr(o, repr);
    int res = _DebugPrint(lvl, buff);
    Py_XDECREF(repr);
    return res;
}

/******************************** CAPSULE FILLING ****************************************/

PyObject *_WrapPointer( void *ptr, const char* name) {
    PyObject *link_cap = PyCapsule_New(ptr, name, NULL);
//    _DebugPrint(4, "Attaching pointer (%p) to capsule (%s)", ptr, name);
    if (link_cap == NULL) {
        PyErr_SetString(PyExc_TypeError, "couldn't create capsule object");
        return NULL;
    } else if (!PyCapsule_IsValid(link_cap, name)) {
        PyErr_SetString(PyExc_ValueError, "couldn't add pointer to invalid capsule object");
        Py_XDECREF(link_cap);
        return NULL;
    }
    return link_cap;
}

// is this even useful...
//void *_ExtractPointer( PyObject *link_cap, const char* name ) {
//    return PyCapsule_GetPointer(link_cap, name);
//}

/******************************** NUMPY MANIPULATION *************************************/

PyObject *_ArrayAsType(PyObject *array, const char *type) {

    PyObject *array_module = PyImport_ImportModule("numpy");
    CHECKNULL(array_module)
    PyObject *astype = PyObject_GetAttrString(array, "astype");
    CHECKCLEAN(astype, array_module)
    PyObject *targetType = PyObject_GetAttrString(array_module, type);
    CHECKCLEAN(targetType, astype, array_module)
    PyObject *xArray = PyObject_CallFunction(astype, "O", targetType);
    CHECKCLEAN(xArray, targetType, astype, array_module)

    return xArray;

}

PyObject *_CreateFromNumPy(const char *ctor, PyObject *args, PyObject *kwargs) {

    PyObject *array_module = PyImport_ImportModule("numpy");
    CHECKNULL(array_module);
    PyObject *builder = PyObject_GetAttrString(array_module, ctor);
    CHECKCLEAN(builder, array_module);
    PyObject *cArray = PyObject_Call(builder, args, kwargs);
    CHECKCLEAN(cArray, builder, array_module);
    CLEANUP(builder, array_module);

    return cArray;
}
PyObject *_CreateFromNumPy(const char *ctor, PyObject *args, const char *dtype) {

    PyObject *array_module = PyImport_ImportModule("numpy");
    CHECKNULL(array_module);
    PyObject *typeObj = PyObject_GetAttrString(array_module, dtype);
    CHECKCLEAN(typeObj, array_module);
    PyObject *kwargs = Py_BuildValue("{s:O}", "dtype", typeObj);
    CHECKCLEAN(kwargs, typeObj, array_module);

    PyObject *cArray = _CreateFromNumPy(ctor, args, kwargs);
    CHECKCLEAN(cArray, kwargs, typeObj, array_module);
    CLEANUP(kwargs, typeObj, array_module);

    return cArray;
}

PyObject *_CreateIdentity(int m, const char *dtype) {

    PyObject *args = Py_BuildValue("(i)", m);
    CHECKNULL(args);
    PyObject *cArray = _CreateFromNumPy("eye", args, dtype);
    CHECKCLEAN(cArray, args);
    CLEANUP(args);

    return cArray;
}
PyObject *_CreateIdentity(int m) {
    return _CreateIdentity(m, "int8");
}
PyObject *_CreateArray(int depth, int *dims, const char *ctor, const char *dtype) {

    PyObject *dimObj = PyList_New(depth);
    CHECKNULL(dimObj);
    for (int j = 0; j<depth; j++){
        PyList_SetItem(dimObj, j, Py_BuildValue("i", dims[j]));
    }
    PyObject *args = Py_BuildValue("(O)", dimObj);
    PyObject *cArray = _CreateFromNumPy(ctor, args, dtype);
    CHECKCLEAN(cArray, args, dimObj);
    CLEANUP(args, dimObj);

    return cArray;
}
PyObject *_CreateArray(int depth, int *dims, const char *ctor) {
    return _CreateArray(depth, dims, ctor, "float64");
}

/**************************** NUMPY TO C++ ****************************************/

std::vector<long> _GetNumpyShape(PyObject* ndarray) {
    PyObject *shp = PyObject_GetAttrString(ndarray, "shape");
    int nums = PyObject_Length(shp);
//    CHECKCLEAN(nums, shp);
    std::vector<long> shape(nums);
    PyObject *iter = PyObject_GetIter(shp);
    PyObject *item; int i = 0; long n;
    while ( (item = PyIter_Next(iter)) ) {
        n = PyLong_AsLong(item);
        Py_XDECREF(item);
        if (PyErr_Occurred()) { break; }
        shape[i++] = n;
    };

    Py_XDECREF(shp);

    return shape;

};

/***** MISC *****/
// might be worth writing this kinda thing in general...
size_t _idx2D(int i, int j, int n, int m) {
    return m * i + j;
}
size_t _idx3D(int i, int j, int k, int n, int m, int l) {
    return m * l * i + l * j + k;
}