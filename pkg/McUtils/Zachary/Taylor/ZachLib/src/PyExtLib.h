

#ifndef PY_EXT_LIB_H
#define PY_EXT_LIB_H

#include "Python.h"
#include <vector>
#include <string>

/******************************** RESCURSIVE MACROS ********************************/
// Silly stuff for getting recursive macros in C++. Helpful for cleanup work

#pragma once

#define CPPX_CONCAT__( a, b )       a ## b
#define CPPX_CONCAT_( a, b )        CPPX_CONCAT__( a, b )
#define CPPX_CONCAT( a, b )         CPPX_CONCAT_( a, b )

#pragma once

#define CPPX_INVOKE( macro, args ) macro args

#define PP_ARG_N( \
_1, _2, _3, _4, _5, _6, _7, _8, _9,_10, \
_11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
_21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
_31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
_41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
_51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
_61,_62,_63,N,...) N

#define PP_RSEQ_N() \
63,62,61,60, \
59,58,57,56,55,54,53,52,51,50, \
49,48,47,46,45,44,43,42,41,40, \
39,38,37,36,35,34,33,32,31,30, \
29,28,27,26,25,24,23,22,21,20, \
19,18,17,16,15,14,13,12,11,10, \
9,8,7,6,5,4,3,2,1,0

#define CPPX_NARG_(...) CPPX_INVOKE( PP_ARG_N, (__VA_ARGS__) )
#define CPPX_NARGS(...) CPPX_NARG_(__VA_ARGS__,PP_RSEQ_N())

/******************************** MACROS ********************************/
// This is all useful boilerplate
#define FUNCNOARGS(meth) PyObject *meth(PyObject* self)
#define FUNCWITHARGS(meth) PyObject *meth(PyObject* self, PyObject* args)
#define FUNCWITHKWARGS(meth) PyObject *meth(PyObject* self, PyObject* args, PyObject* kwargs) )

#define PARSEARGS(...) if ( !PyArg_ParseTuple(args, __VA_ARGS__) ) return NULL;
#define CHECKERROR() if (PyErr_Occurred()) return NULL;
#define CLEANUP_( a ) Py_XDECREF(a)
#define CLEANUP_1( a ) CLEANUP_( a )
#define CLEANUP_2( a, b ) CLEANUP_1( a ); CLEANUP_( b )
#define CLEANUP_3( a, b, c ) CLEANUP_2( a, b ); CLEANUP_( c )
#define CLEANUP_4( a, b, c, d ) CLEANUP_3( a, b, c ); CLEANUP_( d )
#define CLEANUP_5( a, b, c, d, e ) CLEANUP_4( a, b, c, d ); CLEANUP_( e )
#define CLEANUP_6( a, b, c, d, e, f ) CLEANUP_5( a, b, c, d, e ); CLEANUP_( f )
#define CLEANUP_7( a, b, c, d, e, f, g ) CLEANUP_6( a, b, c, d, e, f ); CLEANUP_( g )
#define CLEANUP_8( a, b, c, d, e, f, g, h ) CLEANUP_7( a, b, c, d, e, f, g ); CLEANUP_( h )
#define CLEANUP(...) CPPX_INVOKE( CPPX_CONCAT( CLEANUP_, CPPX_NARGS( __VA_ARGS__ ) ), ( __VA_ARGS__ ) )
// defines a macro that will check if res is NULL and if so clean up the __VA_ARGS__
#define CHECKNULL(res, ...) if (res == NULL) { return NULL; };
#define CHECKCLEAN(res, ...) if (res == NULL) { CLEANUP(__VA_ARGS__); return NULL; };
#define RETURNINT(pkt) return Py_BuildValue("i", pkt);
#define RETURNFLOAT(pkt) return Py_BuildValue("f", pkt);
#define RETURNSTR(car) return Py_BuildValue("s", car);
#define RETURNPTR(ptr) return PyLong_FromVoidPtr(ptr);
#define RETURNBOOL(boo) if (boo) { Py_RETURN_TRUE; } else { Py_RETURN_FALSE; };
#define THREADED(op) Py_BEGIN_ALLOW_THREADS; op; Py_END_ALLOW_THREADS;

/******************************** FUNCTIONS ********************************/
// These are all the exposed functions in the PyExtLib.cpp file

static int _DEBUG_LEVEL = 0;
int _DebugPrint(int level, const char *fmt, ...);
int _DebugMessage(int level, const char *msg);
int _DebugPrintObject(int lvl, PyObject *o);

// this is currently not the most complete list
Py_buffer _GetDataBuffer(PyObject *data);
PyObject *_ArrayAsType(PyObject *array, const char *type);
PyObject *_CreateFromNumPy(const char *ctor, PyObject *args, PyObject *kwargs);
PyObject *_CreateFromNumPy(const char *ctor, PyObject *args, const char *dtype);
PyObject *_CreateIdentity(int m, const char *dtype);
PyObject *_CreateArray(int depth, int *dims, const char *ctor, const char *dtype);
PyObject *_CreateArray(int depth, int *dims, const char *ctor);

std::vector<long> _GetNumpyShape(PyObject* ndarray);

size_t _idx2D(int i, int j, int n, int m);
size_t _idx3D(int i, int j, int k, int n, int m, int l);

/******************************** TEMPLATES ********************************/
// I guess all templates need to be exposed in the header...?
template<typename T>
T *_GetDataBufferArray(Py_buffer *view) {

    T *c_data;
    if ( view == NULL ) return NULL;
    c_data = (T *) view->buf;
    if (c_data == NULL) {
        PyBuffer_Release(view);
    }
    return c_data;

}

template<typename T>
T *_GetDataArray(PyObject *data) {
    Py_buffer view = _GetDataBuffer(data);
    T *array = _GetDataBufferArray<T>(&view);
    CHECKNULL(array);
    return array;
}

void _CopyDataBuffer(PyObject *data, void *buff, long len, int dsize);
template<typename T>
void _CopyDataBuffer(PyObject *data, void *buff, long len) {
    _CopyDataBuffer(data, buff, len, sizeof(T));
}

template<typename T>
void _SetDataBuffer(PyObject *data, void *buff) {

    Py_buffer view_obj = _GetDataBuffer(data);
    Py_buffer *view = &view_obj;
    view->buf = (T *) buff;

}

// might be worth writing this kinda thing in general...
size_t _idx2D(int i, int j, int n, int m);
size_t _idx3D(int i, int j, int k, int n, int m, int l);

template<typename T, typename D>
T _VectorsFromNumpy(PyObject* ndarray) {

    std::vector<long> shape = _GetNumpyShape(ndarray);
    long prod = 1;
    for ( size_t i = 0; i < shape.size(); i++ ) {
        prod *= shape[i];
    }
    D* buf = _GetDataBuffer(ndarray);
    T vec (buf, sizeof(D)*prod); // basically has to be a vector type
    return vec;

}

template<typename T>
T _VectorsFromNumpy(PyObject* ndarray) {

    return _VectorsFromNumpy<T, double>(ndarray);

}

const char *_GetString( PyObject* s, PyObject *pyStr);
const char *_Repr(PyObject *o, PyObject *repr);
#if PY_MAJOR_VERSION == 3
const char *_GetString( PyObject* s, const char *enc, const char *err, PyObject *pyStr);
#else
const char *_GetString( PyObject* s);
const char *_Repr(PyObject *o);
#endif

#endif // PY_EXT_LIB_H