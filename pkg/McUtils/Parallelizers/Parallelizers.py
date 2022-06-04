"""
Provides a simple framework for unifying different parallelism approaches.
Currently primarily targets multiprocessing and mpi4py, but also should work
with Ray. Dask will require more work unfortunately...
"""

import abc, functools, multiprocessing as mp, typing, uuid, os
import numpy as np, pickle, time

from ..Scaffolding import Logger, NullLogger, ObjectRegistry
from .SharedMemory import SharedObjectManager, SharedMemoryList, SharedMemoryDict

__all__ = [
    "Parallelizer",
    "MultiprocessingParallelizer",
    "MPIParallelizer",
    "SerialNonParallelizer",
    "SendRecieveParallelizer"
]

class CallerContract:
    """
    Provides a structure so that a main process and child process can
    be synchronized in their MPI-like calls
    """

    def __init__(self, calls):
        self.calls = calls
        self.which_call = 0

    def handle_call(self, caller, next_call):
        if next_call != self.calls[self.which_call]:
            raise ValueError("caller {} tried to call {} but expected to call {}".format(
                caller,
                next_call,
                self.calls[self.which_call]
            ))
        self.which_call = (self.which_call + 1) % len(self.calls)

class ChildProcessRuntimeError(RuntimeError):
    ...

class Parallelizer(metaclass=abc.ABCMeta):
    """
    Abstract base class to help manage parallelism.
    Provides the basic API that all parallelizers can be expected
    to conform to.
    Provides effectively the union of operations supported by
    `mp.Pool` and `MPI`.
    There is also the ability to lookup and register 'named'
    parallelizers, since we expect a single program to not
    really use more than one.
    This falls back gracefully to the serial case.
    """

    ####################################################################
    #  CONTEXT MANAGEMENT:
    #   details for initialization/registration/getting and setting
    #   up a parallelizer
    #
    _par_registry = None
    default_printer = print
    def __init__(self, logger=None, contract=None, uid=None):
        self._active_sentinel=0
        self._pickle_prot = None
        if logger is None:
            logger = Logger()
        self.logger=logger
        self.contract=contract if contract is None or isinstance(contract, CallerContract) else CallerContract(contract)
        self._default_stack = None
        self.uid = uuid.uuid1()
        self._pid = None
        # if printer is None:
        #     self._logger = Logger()
        #     self._default_printer = self._logger.log_print
        # else:
        #

    @classmethod
    def load_registry(cls):
        if cls._par_registry is None:
            cls._par_registry = ObjectRegistry(default=SerialNonParallelizer())
        return cls._par_registry
    @property
    def parallelizer_registry(self):
        return self.load_registry()

    @classmethod
    def get_default(cls):
        """
        For compat.

        :return:
        :rtype:
        """
        return cls.lookup(None)
    @classmethod
    def lookup(cls, key):
        """
        Checks in the registry to see if a given parallelizer is there
        otherwise returns a `SerialNonParallelizer`.
        :param key:
        :type key:
        :return:
        :rtype:
        """
        return cls.load_registry().lookup(key)

    def register(self, key):
        """
        Checks in the registry to see if a given parallelizer is there
        otherwise returns a `SerialNonParallelizer`.
        :param key:
        :type key:
        :return:
        :rtype:
        """
        self.parallelizer_registry.register(key, self)

    @property
    def active(self):
        return self._active_sentinel > 0

    @abc.abstractmethod
    def initialize(self):
        """
        Initializes a parallelizer
        if necessary
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def finalize(self, exc_type, exc_val, exc_tb):
        """
        Finalizes a parallelizer (if necessary)
        if necessary
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    def __enter__(self):
        """
        Allows the parallelizer context to be set
        using `with`
        :return:
        :rtype:
        """
        if not self.active:
            self._default_stack = self.load_registry().temp_default(self)
            self._default_stack.__enter__()
            self.register(self.uid)
            self.initialize()
            self._pickle_prot = pickle.DEFAULT_PROTOCOL
            pickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL
        self._active_sentinel += 1
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Allows the parallelizer context to be unset
        using `with`
        :param exc_type:
        :type exc_type:
        :param exc_val:
        :type exc_val:
        :param exc_tb:
        :type exc_tb:
        :return:
        :rtype:
        """
        self._active_sentinel -= 1
        if not self.active:
            if self._default_stack is not None:
                self._default_stack.__exit__(exc_type, exc_val, exc_tb)
                self._default_stack = None
            self.finalize(exc_type, exc_val, exc_tb)
            # self._pickle_prot = pickle.DEFAULT_PROTOCOL
            pickle.DEFAULT_PROTOCOL = self._pickle_prot

    ####################################################################
    #  CONTROL FLOW:
    #   clean ways to manage flow of program
    #
    class InMainProcess:
        """
        Singleton representing being on the
        main process
        """
    class InWorkerProcess:
        """
        Singleton representing being on a
        worker process
        """
    @property
    @abc.abstractmethod
    def on_main(self):
        """
        Returns whether or not the executing process is the main
        process or not
        :return:
        :rtype:
        """

    @staticmethod
    def main_restricted(func):
        """
        A decorator to indicate that a function should only be
        run when on the main process
        :param func:
        :type func:
        :return:
        :rtype:
        """

        def main_process_func(*args, parallelizer=None, **kwargs):
            if parallelizer is None:
                parallelizer = Parallelizer.lookup(None)
            if parallelizer.on_main:
                kwargs['parallelizer'] = parallelizer
                return func(*args, **kwargs)
            else:
                return Parallelizer.InWorkerProcess

        return main_process_func

    @staticmethod
    def worker_restricted(func):
        """
        A decorator to indicate that a function should only be
        run when on a worker process
        :param func:
        :type func:
        :return:
        :rtype:
        """

        @functools.wraps(func)
        def worker_process_func(*args, parallelizer=None, **kwargs):
            if parallelizer is None:
                parallelizer = Parallelizer.lookup(None)
            if not parallelizer.on_main:
                kwargs['parallelizer'] = parallelizer
                return func(*args, **kwargs)
            else:
                return Parallelizer.InMainProcess

        return worker_process_func

    ####################################################################
    #  MPI API:
    #   Require core methods to be implemented to support the
    #   high-level MPI api.
    #   Currently only blocking operations are expected to be supported,
    #   since it's unlikely we'll ever have a use case for async calls
    #
    @abc.abstractmethod
    def send(self, data, loc, **kwargs):
        """
        Sends data to the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def receive(self, data, loc, **kwargs):
        """
        Receives data from the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def broadcast(self, data, **kwargs):
        """
        Sends the same data to all processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def scatter(self, data, **kwargs):
        """
        Performs a scatter of data to the different
        available parallelizer processes.
        *NOTE:* unlike in the MPI case, `data` does not
        need to be evenly divisible by the number of available
        processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def gather(self, data, **kwargs):
        """
        Performs a gather of data from the different
        available parallelizer processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    ####################################################################
    #  POOL API:
    #   require core methods to be implemented to act like an `mp.Pool`
    #
    @abc.abstractmethod
    def map(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    @abc.abstractmethod
    def starmap(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map with unpacking of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        raise NotImplementedError("Parallelizer is an abstract base class")

    @abc.abstractmethod
    def apply(self, func, *args, main_kwargs=None, **kwargs):
        """
        Runs the callable `func` in parallel
        :param func:
        :type func:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    def run(self, func, *args, comm=None, main_kwargs=None, **kwargs):
        """
        Calls `apply`, but makes sure state is handled cleanly

        :param func:
        :type func:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        with self:
            return self.apply(func, *args, comm=comm, main_kwargs=main_kwargs, **kwargs)

    mode_map = {}
    @classmethod
    def from_config(cls,
                    mode=None,
                    **kwargs
                    ):
        if mode not in cls.mode_map:
            raise KeyError("don't know what to do with parallelization mode '{}'".format(
                mode
            ))
        par_cls = cls.mode_map[mode]
        return par_cls.from_config(**kwargs)

    ####################################################################
    #  UTILITY API:
    #   necessary things to make API more useable
    #
    @property
    def nprocs(self):
        """
        Returns the number of processes the parallelizer has
        to work with
        :return:
        :rtype:
        """
        return self.get_nprocs()
    @abc.abstractmethod
    def get_nprocs(self):
        """
        Returns the number of processes
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @property
    def id(self):
        """
        Returns some form of identifier for the current process
        :return:
        :rtype:
        """
        return self.get_id()
    @property
    def pid(self):
        if self._pid is None:
            self._pid = os.getpid()
        return self._pid
    @abc.abstractmethod
    def get_id(self):
        """
        Returns the id for the current process
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    @property
    def printer(self):
        if self.logger is None:
            return self.default_printer
        else:
            return self.logger.log_print
    @printer.setter
    def printer(self, p):
        self._printer = p
    def main_print(self, *args, **kwargs):
        """
        Prints from the main process
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        self.printer(*args, **kwargs)
    def worker_print(self, *args, **kwargs):
        """
        Prints from a main worker process
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        self.printer(" ".join(["On Worker {} ({}):".format(self.id, self.pid), *(str(x) for x in args)]), **kwargs)
    def print(self, *args, where='both', **kwargs):
        """
        An implementation of print that operates differently on workers than on main
        processes

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.on_main:
            if where in {'both', 'main'}:
                return self.main_print(*args, **kwargs)
        else:
            if where in {'both', 'worker'}:
                return self.worker_print(*args, **kwargs)

    @abc.abstractmethod
    def wait(self):
        """
        Causes all processes to wait until they've met up at this point.
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    def __repr__(self):
        try:
            id = self.id
        except:
            id = None
        try:
            nprocs = self.nprocs
        except:
            nprocs = None
        return "{}(id={}, nprocs={}, uuid={})".format(type(self).__name__, id, nprocs, self.uid)

    def share(self, obj):
        """
        Converts `obj` into a form that can be cleanly used with shared memory via a `SharedObjectManager`

        :param obj:
        :type obj:
        :return:
        :rtype:
        """

        if isinstance(obj, dict):
            sharer = SharedMemoryDict(obj, parallelizer=self)
        elif isinstance(obj, (list, tuple)):
            sharer = SharedMemoryList(obj, parallelizer=self)
        else:
            sharer = SharedObjectManager(obj, parallelizer=self)
            sharer.share()

        return sharer

class SendRecieveParallelizer(Parallelizer):
    """
    Parallelizer that implements `scatter`, `gather`, `broadcast`, and `map`
    based on just having a communicator that supports `send` and `receive methods
    """

    class ReceivedError:
        def __init__(self, error):
            self.error = error
    class SendReceieveCommunicator(metaclass=abc.ABCMeta):
        """
        A base class that provides an interface for
        sending/receiving data from a specific subprocesses
        """
        @property
        @abc.abstractmethod
        def locations(self):
            """
            Returns the list of locations known by the
            `SendReceieverCommunicator`
            :return:
            :rtype:
            """
        @property
        @abc.abstractmethod
        def location(self):
            """
            Returns the _current_ location
            :return:
            :rtype:
            """
        @abc.abstractmethod
        def send(self, data, loc, **kwargs):
            """
            Sends the specified data to loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
        @abc.abstractmethod
        def receive(self, data, loc, **kwargs):
            """
            Receives the specified data from loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
    @property
    @abc.abstractmethod
    def comm(self):
        """
        Returns the communicator used by the paralellizer
        :return:
        :rtype: SendRecieveParallelizer.SendReceieveCommunicator
        """
    def send(self, data, loc, **kwargs):
        """
        Sends data to the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.contract is not None:
            self.contract.handle_call(self, "send")
        return self.comm.send(data, loc, **kwargs)
    def receive(self, data, loc, **kwargs):
        """
        Receives data from the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.contract is not None:
            self.contract.handle_call(self, "receive")
        val = self.comm.receive(data, loc, **kwargs)
        if isinstance(val, self.ReceivedError):
            raise val.error
        else:
            return val
    def broadcast(self, data, **kwargs):
        """
        Sends the same data to all processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.contract is not None:
            self.contract.handle_call(self, "broadcast")
        if self.on_main:
            for i in self.comm.locations[1:]:
                self.comm.send(data, i, **kwargs)
            return data
        else:
            return self.comm.send(data, self.comm.location) # effectively a receive...
    def scatter(self, data, **kwargs):
        """
        Performs a scatter of data to the different
        available parallelizer processes.
        *NOTE:* unlike in the MPI case, `data` does not
        need to be evenly divisible by the number of available
        processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if self.contract is not None:
            self.contract.handle_call(self, "scatter")
        if self.on_main:
            locs = list(self.comm.locations) # gotta be safe
            nlocs = len(locs)
            chunk_size = len(data) // nlocs
            chunk_sizes = [chunk_size] * nlocs
            chunk_coverage = (chunk_size*nlocs)
            for i in range(len(data)-chunk_coverage):
                chunk_sizes[i] += 1
            s = chunk_sizes[0]
            main_data = data[:s]
            for i, b in zip(locs[1:], chunk_sizes[1:]):
                chunk = data[s:s+b]
                self.comm.send(chunk, i, **kwargs)
                s+=b
            return main_data
        else:
            return self.comm.send(data, self.comm.location) # effectively a receive...
    def gather(self, data, **kwargs):
        """
        Performs a gather of data from the different
        available parallelizer processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if self.contract is not None:
            self.contract.handle_call(self, "gather")
        if self.on_main:
            locs = list(self.comm.locations) # gotta be safe
            nlocs = len(locs) # we reserve space for the main thread
            recv = [None] * nlocs
            recv[0] = data
            all_numpy = True
            for n, i in enumerate(locs[1:]):
                res = self.comm.receive(data, i, **kwargs)
                if not isinstance(res, np.ndarray):
                    all_numpy=False
                if isinstance(res, Exception):
                    raise res
                recv[n+1] = res
            if all_numpy:
                # special case
                try:
                    recv = np.concatenate(recv, axis=0)
                except:
                    pass
            return recv
        else:
            return self.comm.receive(data, self.comm.location, **kwargs) # effectively a send...
    def map(self, func, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        # self.wait()
        self.print("Scattering Data", log_level=Logger.LogLevel.MoreDebug)
        data = self.scatter(data, **kwargs)
        # self.wait()
        self.print("Broadcasting Extra Args", log_level=Logger.LogLevel.MoreDebug)
        extra_args = self.broadcast(extra_args, **kwargs)
        # self.wait()
        self.print("Broadcasting Extra Kwargs", log_level=Logger.LogLevel.MoreDebug)
        extra_kwargs = self.broadcast(extra_kwargs, **kwargs)
        if extra_args is None:
            extra_args = ()
        if extra_kwargs is None:
            extra_kwargs = {}
        # try:
        evals = [func(sub_data, *extra_args, **extra_kwargs) for sub_data in data]
        # except Exception as e:
        #     self.gather(e, **kwargs)
        #     raise
        res = self.gather(evals, **kwargs)
        if self.on_main:
            return sum(res, [])
        else:
            return Parallelizer.InWorkerProcess
    def starmap(self, func, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map with unpacking of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        data = self.scatter(data, **kwargs)
        extra_args = self.broadcast(extra_args)
        extra_kwargs = self.broadcast(extra_kwargs)
        if extra_args is None:
            extra_args = ()
        if extra_kwargs is None:
            extra_kwargs = {}
        # try:
        evals = [func(*sub_data, *extra_args, **extra_kwargs) for sub_data in data]
        # except Exception as e:
        #     self.gather(e, **kwargs)
        #     raise
        res = self.gather(evals, **kwargs)
        if self.on_main:
            return sum(res, [])
        else:
            return Parallelizer.InWorkerProcess

    def wait(self):
        """
        Causes all processes to wait until they've met up at this point.
        :return:
        :rtype:
        """

        # flag = self.broadcast("SyncFlag")
        self.gather("SyncFlag") # sends data to main
        self.broadcast("SyncFlag")

class MultiprocessingParallelizer(SendRecieveParallelizer):
    """
    Parallelizes using a  process pool and a runner
    function that represents a "main loop".
    """

    class SendRecvQueuePair:
        def __init__(self, id:int, manager:'mp.managers.SyncManager'):
            self.id=id
            self.send_queue = manager.Queue()
            self.receive_queue = manager.Queue()
            self.init_flag = manager.Event()
        def __repr__(self):
            return "{}({})".format(type(self).__name__, self.id)
    class PoolCommunicator(SendRecieveParallelizer.SendReceieveCommunicator):
        """
        Defines a serializable process communicator
        that allows communication with a managed `mp.Pool`
        to support `send` and `receive` and therefore to
        support the rest of the necessary bits of the MPI API
        """
        def __init__(self,
                     parent: 'MultiprocessingParallelizer',
                     id:int,
                     queues: typing.Iterable['MultiprocessingParallelizer.SendRecvQueuePair'],
                     initialization_timeout:float=None,
                     group:typing.Iterable['MultiprocessingParallelizer.PoolCommunicator']=None
                     ):
            self.parent = parent
            self.id = id
            self.queues = tuple(queues)
            self.initialization_timeout=initialization_timeout
            self.group = tuple(group) if group is not None else None

        def __repr__(self):
            return "{}({})".format(
                type(self).__name__,
                self.id
            )

        class PoolError(Exception):
            """
            For errors arising from Pool operations
            """
            pass

        def initialize(self):
            """
            Performs initialization of the communicator
            (basically just waits until all threads say all is well)
            :return:
            :rtype:
            """
            if not self.queues[self.id].init_flag.is_set():
                self.parent.print("setting init flag on {id}", id=self.id, log_level=Logger.LogLevel.MoreDebug)
                self.queues[self.id].init_flag.set()
            if self.parent.on_main:
                for i, q in enumerate(self.queues):
                    if not q.init_flag.is_set():
                        self.parent.print("checking init flag on {i}".format(i=i), log_level=Logger.LogLevel.MoreDebug)
                        wat = q.init_flag.wait(self.initialization_timeout)
                        if not wat:
                            raise self.PoolError("Failed to initialize pool")

        def reset(self):
            """
            Performs initialization of the communicator
            (basically just waits until all threads say all is well)
            :return:
            :rtype:
            """
            for i, q in enumerate(self.queues):
                q.init_flag.clear()
            # if not self.queues[self.id].init_flag.is_set():
            #     self.parent.print("setting init flag on {id}", id=self.id, log_level=Logger.LogLevel.MoreDebug)
            #     self.queues[self.id].init_flag.set()
            # if self.parent.on_main:
            #     for i, q in enumerate(self.queues):
            #         if not q.init_flag.is_set():
            #             self.parent.print("checking init flag on {i}".format(i=i), log_level=Logger.LogLevel.MoreDebug)
            #             wat = q.init_flag.wait(self.initialization_timeout)
            #             if not wat:
            #                 raise self.PoolError("Failed to initialize pool")

        @property
        def locations(self):
            """
            Returns the list of queue ids
            :return:
            :rtype:
            """
            return [x.id for x in self.queues]
        @property
        def location(self):
            """
            Returns the _current_ location
            :return:
            :rtype:
            """
            return self.id

        def send(self, data, loc, **kwargs):
            """
            Sends the specified data to loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """

            queue = self.queues[loc].send_queue #type: mp.queues.Queue
            if loc == self.id:
                self.parent.print("Send: getting on {id}".format(id=self.id), log_level=Logger.LogLevel.MoreDebug)
                res = queue.get()
                self.parent.print("Send: got on {id}".format(id=self.id), log_level=Logger.LogLevel.MoreDebug)
                res = pickle.loads(res)
                return res
            else:
                self.parent.print("Send: putting {id} to {loc}".format(id=self.id, loc=loc), log_level=Logger.LogLevel.MoreDebug)
                data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                queue.put(data)
                self.parent.print("Send: put on {id} to {loc}".format(id=self.id, loc=loc), log_level=Logger.LogLevel.MoreDebug)
                return data
        def receive(self, data, loc, **kwargs):
            """
            Receives the specified data from loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            queue = self.queues[loc].receive_queue
            if loc != self.id:
                self.parent.print("Recv: getting on {id} from {loc}".format(id=self.id, loc=loc), log_level=Logger.LogLevel.MoreDebug)
                res = queue.get()
                res = pickle.loads(res)
                self.parent.print("Recv: got on {id} from {loc}".format(id=self.id, loc=loc), log_level=Logger.LogLevel.MoreDebug)
                return res
            else:
                self.parent.print("Recv: putting on {id}".format(id=self.id), log_level=Logger.LogLevel.MoreDebug)
                data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                queue.put(data)
                self.parent.print("Recv: put on {id}".format(id=self.id), log_level=Logger.LogLevel.MoreDebug)
                return data

        def get_subcomm(self, idx):
            return type(self)(
                self.parent,
                self.id,
                tuple(self.queues[i] for i in idx),
                initialization_timeout=self.initialization_timeout,
                group=None if self.group is None else tuple(self.group[i] for i in idx)
            )

    _is_worker=False # global flag to be overridden
    def __init__(self,
                 worker=False,
                 pool:mp.Pool=None,
                 context=None,
                 manager=None,
                 logger=None,
                 contract=None,
                 comm=None,
                 rank=None,
                 allow_restart=True,
                 initialization_timeout=.5,
                 **kwargs
                 ):
        self.initialization_timeout=initialization_timeout
        super().__init__(logger=logger, contract=contract)
        self.opts=kwargs
        self.pool=pool
        self.worker=worker
        self.ctx=context
        self.manager=manager
        self._comm = comm
        self._id = rank
        self.nproc = None
        self.allow_restart = allow_restart

    def get_nprocs(self):
        return self.nproc
    def get_id(self):
        if self._id is None:
            return self.comm.id
        else:
            return self._id

    @property
    def comm(self):
        """
        Returns the communicator used by the paralellizer
        :return:
        :rtype: MultiprocessingParallelizer.PoolCommunicator
        """
        if self._comm is None:
            self.logger.log_print("initializing PoolCommunicators...", log_level=self.logger.LogLevel.MoreDebug)
            comm_list = [
                self.PoolCommunicator(self, i, self.queues, initialization_timeout=self.initialization_timeout) for
                i in range(0, self.nproc)
            ]
            self.logger.log_print("got comm group {g}", g=comm_list, log_level=self.logger.LogLevel.MoreDebug)
            self._comm = comm_list[0]
            self._comm.group = comm_list
        return self._comm
    @comm.setter
    def comm(self, c):
        self._comm = c

    # def to_state(self, serializer=None):
    #     return self.__getstate__()
    # @classmethod
    # def from_state(cls, state, serializer=None):
    #     return cls(
    #         worker=state['worker']
    #     )
    def __getstate__(self):
        # most things don't need to be mapped over...
        state = self.__dict__.copy()
        state['pool'] = None
        state['worker'] = True
        state['manager'] = None
        state['_default_stack'] = None
        # state['comm'] = None
        state['_comm'] = None
        state['queues'] = None
        state['_pid'] = None
        # state['_active_sentinel'] = 0
        # state['_id'] = self.id
        # state['_par_registry'] = None
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # allow for better syncing...
        if self.uid in self.parallelizer_registry:
            parent = self.lookup(self.uid)
            self.__dict__.update(parent.__dict__)
            # print("?", parent)
        # else:
        #     print(":o", self, list(self.parallelizer_registry.values()))
        #     self.register(self.uid)

    @staticmethod
    def _run(runner, comm:PoolCommunicator, args, kwargs, main_kwargs=None):
        """
        Static runner function that just dispatches methods out to
        cores
        :param runner:
        :type runner:
        :param comm:
        :type comm:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if runner is None: #quick ignore
            return None

        self = comm.parent
        self._comm = comm # makes a cyclic dependency...not sure how best to fix that
        self.worker = comm.id > 0
        with self:
            # print(self._active_sentinel, list(self.parallelizer_registry.values()))
            if self.on_main:
                self.print(
                    "Starting Parallelizer with {runner} over processor group {grp}".format(
                        runner=runner,
                        grp=self.comm.locations
                    ),
                    log_level=Logger.LogLevel.MoreDebug
                )
            else:
                self.print(
                    "starting process {p} at {t}".format(
                        p=self.pid,
                        t=time.ctime()
                    ),
                    log_level=Logger.LogLevel.MoreDebug
                )
            self._comm.initialize()
            kwargs['parallelizer'] = comm.parent
            if main_kwargs is None:
                main_kwargs = {}
            if self.on_main:
                return runner(*args, **main_kwargs, **kwargs)
            else:
                try:
                    return runner(*args, **main_kwargs, **kwargs)
                except Exception as e:
                    import traceback as tb
                    self.print(tb.format_exc())
                    comm.send(self.ReceivedError(e), 0)
                    raise

    def apply(self, func, *args, comm=None, main_kwargs=None, **kwargs):
        """
        Applies func to args in parallel on all of the processes

        :param func:
        :type func:
        :param args:
        :type args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        import multiprocessing as mp

        if comm is None:
            comm = self.comm
        elif not isinstance(comm, SendRecieveParallelizer.SendReceieveCommunicator):
            comm = self.comm.get_subcomm(comm)

        _comm = self.comm
        try:
            self.comm = comm

            # since comm might be smaller than the pool
            # we need to send "ignore" flags to those pools
            if comm.group is None:
                raise ValueError("{} needs to have a processor group to parallelize over".format(
                    comm
                ))
            group_map = {g.id:g for g in comm.group}
            if 0 not in group_map:
                raise ValueError("{} needs to have the main process in it as designed".format(
                    comm
                ))

            mapping = [
                [
                    func,
                    group_map[i],
                    args,
                    kwargs
                ] if i in group_map else [
                    None,
                    None,
                    None,
                    None
                ] for i in range(1, self.nproc)
            ]
            pool = self.pool #type: mp.pool.Pool
            self.comm.reset()
            subsidiary = pool.starmap_async(self._run, mapping)
            # pool._worker_handler.join()
            # pool._task_handler.join()
            # pool._help_stuff_finish(
            #     pool._inqueue,
            #
            # )
            # while not (
            #          pool._taskqueue.empty()
            #          and pool._inqueue.empty()
            # ):
            #     time.sleep(.05)
            # if self.initialization_timeout is not None:
            #     time.sleep(self.initialization_timeout)
            try:
                main = self._run(func, comm, args, kwargs, main_kwargs=main_kwargs)
            except self.PoolCommunicator.PoolError:
                # check for errors on subsidiary...
                subsidiary.get(timeout=self.initialization_timeout)
                raise
            subs = subsidiary.get() # just to effect a wait
        finally:
            self.comm = _comm
        return main

    def _get_pool(self,
                  manager: mp.Manager,
                  **kwargs
                  ):
        if 'processes' not in kwargs:
            kwargs['processes'] = mp.cpu_count() - 1
        if 'initializer' not in kwargs:
            kwargs['initializer'] = self._set_is_worker
        return manager.Pool(**kwargs)
    @staticmethod
    def get_pool_context(pool):
        return pool._ctx # I don't like doing this but seems like only way?
    @staticmethod
    def get_pool_nprocs(pool):
        return pool._processes  # see above

    def _reset_mp_caches(self):
        self.pool = None
        self.queues = None
        self.comm = None
    @classmethod
    def _set_is_worker(cls, *args):
        """
        Sets a flag so that worker processes
        can know immediately that they are workers
        """
        cls._is_worker = True
    def initialize(self, allow_restart=None):
        if not self.worker:
            if self.pool is None:
                self.print("Initializing pool...", log_level=Logger.LogLevel.MoreDebug)
                if self.ctx is None:
                    self.ctx = mp.get_context() # get the default context
                self.pool = self._get_pool(self.ctx, **self.opts)
            elif self.ctx is None:
                self.ctx = self.get_pool_context(self.pool)
            if self.manager is None:
                self.manager = mp.Manager()
            allow_restart = self.allow_restart if allow_restart is None else allow_restart
            if allow_restart:
                try:
                    self.pool.__enter__()
                except ValueError:
                    # fix poos
                    self._reset_mp_caches()
                    return self.initialize(allow_restart=False)
            else:
                self.pool.__enter__()
            self.nproc = self.get_pool_nprocs(self.pool)
            self.pool.map(self._set_is_worker, [None] * self.nproc)
            # just to be safe
            self._is_worker = False
            self.worker = False
            self.queues = [self.SendRecvQueuePair(i, self.manager) for i in range(0, self.nproc)]

    def finalize(self, exc_type, exc_val, exc_tb):
        if not self.worker:
            if self.pool is not None:
                self.pool.__exit__(exc_type, exc_val, exc_tb)
            self.queues = None
            self._comm = None
    @property
    def on_main(self):
        return not self.worker

    @classmethod
    def from_config(cls, **kw):
        return cls(**kw)
Parallelizer.mode_map['multiprocessing'] = MultiprocessingParallelizer

class MPIParallelizer(SendRecieveParallelizer):
    """
    Parallelizes using `mpi4py`
    """

    class MPICommunicator(SendRecieveParallelizer.SendReceieveCommunicator):
        """
        A base class that provides an interface for
        sending/receiving data from a specific subprocesses
        """

        def __init__(self, parent, mpi_comm, api):
            self.parent = parent
            self.comm = mpi_comm
            self.api = api
        @property
        def type_map(self):
            """
            Kinda hacky type map...hopefully
            sufficient to just have a few core numeric types?
            """
            return {
                'int32': self.api.INT,
                'int64': self.api.LONG,
                'float32': self.api.FLOAT,
                'float64': self.api.DOUBLE
            }
        def get_mpi_type(self, dtype):
            """
            Gets the MPI datatype for the numpy
            dtype
            """
            tm = self.type_map
            key = dtype.name
            if key not in tm:
                raise KeyError(
                    "mapping from {} to MPI datatype not included...".format(dtype)
                    )
            return tm[key]
        @property
        def nprocs(self):
            """
            Returns the number of available processes
            :return:
            :rtype:
            """
            return self.comm.Get_size()
        @property
        def locations(self):
            """
            Returns the list of locations known by the
            `SendReceieverCommunicator`
            :return:
            :rtype:
            """
            return range(self.comm.Get_size())
        @property
        def location(self):
            """
            Returns the _current_ location
            :return:
            :rtype:
            """
            return self.comm.Get_rank()
        def send(self, data, loc, **kwargs):
            """
            Sends the specified data to loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if loc == self.location:
                return self.comm.recv(source=self.api.ANY_SOURCE)
            else:
                self.comm.send(data, dest=loc)
        def receive(self, data, loc, root=0, **kwargs):
            """
            Receives the specified data from loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if loc != self.location:
                self.comm.send(self.location, dest=loc)
                return self.comm.recv(source=loc)
            else:
                where_to = self.comm.recv(source=self.api.ANY_SOURCE)
                self.comm.send(data, dest=where_to)
        def broadcast(self, data, root=0, **kwargs):
            """
            Sends the same data to all processes

            :param data:
            :type data:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if isinstance(data, np.ndarray):
                return self.comm.Bcast(data, root=root)
            else:
                return self.comm.bcast(data, root=root)
        def scatter_obj(self, data, root=0, **kwargs):
            """
            Scatters data to the different MPI ranks using a series
            of send calls.
            This is the default for anything except numpy arrays.

            :param data:
            :type data:
            :param root:
            :type root:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if self.location == root:
                locs = list(self.locations)  # gotta be safe
                nlocs = len(locs)
                chunk_size = len(data) // nlocs
                chunk_sizes = [chunk_size] * nlocs
                for i in range(len(data) - (chunk_size*nlocs)):
                    chunk_sizes[i] += 1

                s = chunk_sizes[0]
                main_data = data[:s]
                for i, b in zip(locs[1:], chunk_sizes[1:]):
                    chunk = data[s:s + b]
                    self.send(chunk, i, **kwargs)
                    s += b
                return main_data
        def scatter(self, data, root=0, shape=None, dtype=None, **kwargs):
            """
            Performs a scatter of data to the different
            available parallelizer processes.
            *NOTE:* unlike in the MPI case, `data` does not
            need to be evenly divisible by the number of available
            processes

            :param data:
            :type data:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """

            if isinstance(data, np.ndarray) or data is None:
                if root == self.location and data is None:
                    raise TypeError("'None' is not scatterable. Try `broadcast`.")
                if root == self.location:
                    send_buf = np.ascontiguousarray(data)
                else:
                    send_buf = None
                if shape is None:
                    if root == self.location:
                        shape = data.shape
                    shape = self.broadcast(shape)
                if dtype is None:
                    if root == self.location:
                        dtype = data.dtype
                    dtype = self.broadcast(dtype)
                np.empty(shape, dtype=dtype)
                ranks = self.comm.Get_size()
                ndat = shape[0]
                block_size = ndat // ranks
                block_remainder = ndat - (block_size*ranks)
                if block_remainder == 0:
                    shape = (block_size,) + shape[1:]
                    recv_buf = np.empty(shape, dtype=dtype)
                    self.comm.Scatter(send_buf, recv_buf, root=root)
                    return recv_buf
                else:
                    # it turns out MPI4Py has only an unsophisticated
                    # implementation of Scatterv that explicitly needs the
                    # appropriate block offsets, since it basically expects
                    # to have a flattened form of the array (just like MPI)
                    block_sizes = [block_size]*ranks
                    for i in range(block_remainder):
                        block_sizes[i] += 1

                    recv_buf = np.empty((block_sizes[self.location],) + shape[1:], dtype=dtype)

                    block_offset = int(np.prod(shape[1:]))
                    block_sizes = np.array(block_sizes)*block_offset
                    block_offsets = np.concatenate([[0], np.cumsum(block_sizes)])[:-1]
                    # print(block_offsets, block_offset, block_sizes)

                    # self.parent.print("block sizes: {} offsets: {}".format(block_sizes, block_offsets))
                    self.comm.Scatterv(
                        [
                            send_buf,
                            block_sizes,
                            block_offsets,
                            self.get_mpi_type(dtype)
                        ],
                        recv_buf,
                        root=root
                    )
                    self.parent.print("sending", send_buf, recv_buf, self.get_mpi_type(dtype).name, log_level=Logger.LogLevel.MoreDebug)
                    return recv_buf
            else:
                return self.scatter_obj(data, root=root, **kwargs)
        def gather_obj(self, data, root=0, **kwargs):
            if self.location == root:
                locs = list(self.locations)  # gotta be safe
                nlocs = len(locs)  # we reserve space for the main thread
                recv = [None] * nlocs
                recv[0] = data
                for n, i in enumerate(locs[1:]):
                    recv[n + 1] = self.receive(data, i, **kwargs)
                return recv
            else:
                return self.receive(data, self.location, **kwargs)  # effectively a send...
        def gather(self, data, root=0, shape=None, dtype=None, **kwargs):
            """
            Performs a gather from the different
            available parallelizer processes.

            :param data:
            :type data:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """

            if isinstance(data, np.ndarray):
                # make sure everyone knows what's up
                send_buf = np.ascontiguousarray(data)
                if shape is None:
                    block_sizes = self.gather(data.shape[0], root=root)
                    if block_sizes is not None:
                        ndat = int(np.sum(block_sizes))
                        if root == self.location:
                            shape = (ndat,) + data.shape[1:]
                    shape = self.broadcast(shape)
                if dtype is None:
                    if root == self.location:
                        dtype = data.dtype
                    dtype = self.broadcast(dtype)
                # otherwise send shit around
                ranks = self.comm.Get_size()
                ndat = shape[0]
                block_size = ndat // ranks
                block_remainder = ndat - (block_size*ranks)
                if block_remainder == 0:
                    if root == self.location:
                        recv_buf = np.empty(shape, dtype=dtype)
                    else:
                        recv_buf = None
                    self.comm.Gather(send_buf, recv_buf, root=root)
                    return recv_buf
                else:
                    block_sizes = [block_size] * ranks
                    for i in range(block_remainder):
                        block_sizes[i] += 1
                    block_offset = int(np.prod(shape[1:]))
                    block_sizes = np.array(block_sizes)*block_offset
                    block_offsets = np.concatenate([[0], np.cumsum(block_sizes)])[:-1]
                    if root == self.location:
                        recv_buf = np.empty(shape, dtype=dtype)
                    else:
                        recv_buf = None
                    self.comm.Gatherv(
                        send_buf,
                        [
                            recv_buf,
                            block_sizes,
                            block_offsets,
                            self.get_mpi_type(dtype)
                        ],
                        root=root
                    )

                    return recv_buf
            else:
                return self.gather_obj(data, root=root, **kwargs)

    def __init__(self, root=0, comm=None, contract=None, logger=None):
        super().__init__(contract=contract, logger=logger)

        from mpi4py import MPI as api
        self.api = api

        if comm is None:
            comm = api.COMM_WORLD
        self.root = root
        self._comm = self.MPICommunicator(self, comm, api)
        self.world_size = None
        self.world_rank = None

    def get_nprocs(self):
        return self._comm.nprocs
    def get_id(self):
        return self._comm.location

    def initialize(self):
        # handled by mpi4py?
        pass
    def finalize(self, exc_type, exc_val, exc_tb):
        # handled by mpi4py?
        pass

    @property
    def comm(self):
        """
        Returns the communicator used by the paralellizer
        :return:
        :rtype: MPIParallelizer.MPICommunicator
        """
        return self._comm

    @property
    def on_main(self):
        return self.comm.location == 0

    def broadcast(self, data, **kwargs):
        """
        Sends the same data to all processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if self.contract is not None:
            self.contract.handle_call(self, "broadcast")
        return self.comm.broadcast(data, root=self.root, **kwargs)
    def scatter(self, data, shape=None, **kwargs):
        """
        Performs a scatter of data to the different
        available parallelizer processes.
        *NOTE:* unlike in the MPI case, `data` does not
        need to be evenly divisible by the number of available
        processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if self.contract is not None:
            self.contract.handle_call(self, "scatter")
        return self.comm.scatter(data, root=self.root, shape=shape, **kwargs)
    def gather(self, data, shape=None, **kwargs):
        """
        Performs a gather of data from the different
        available parallelizer processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if self.contract is not None:
            self.contract.handle_call(self, "gather")
        return self.comm.gather(data, root=self.root, shape=shape, **kwargs)
    def map(self, func, data, input_shape=None, output_shape=None, **kwargs):
        """
        Performs a parallel map of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        sub_data = self.scatter(data, shape=input_shape, **kwargs)
        res = func(sub_data)
        return self.gather(res, shape=output_shape, **kwargs)

    def apply(self, func, *args, **kwargs):
        """
        Applies func to args in parallel on all of the processes.
        For MPI, since jobs are always started with mpirun, this
        is just a regular apply

        :param func:
        :type func:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return func(*args, parallelizer=self, **kwargs)

    @classmethod
    def from_config(cls, **kw):
        return cls(**kw)
Parallelizer.mode_map['mpi'] = MPIParallelizer

class SerialNonParallelizer(Parallelizer):
    """
    Totally serial evaluation for cases where no parallelism
    is provide
    """

    def get_nprocs(self):
        return 1
    def get_id(self):
        return 0

    def initialize(self):
        """
        Initializes a parallelizer
        if necessary
        :return:
        :rtype:
        """
        pass
    def finalize(self, exc_type, exc_val, exc_tb):
        """
        Finalizes a parallelizer (if necessary)
        if necessary
        :return:
        :rtype:
        """
        pass
    @property
    def on_main(self):
        """
        Returns whether or not the executing process is the main
        process or not
        :return:
        :rtype:
        """
        return True
    def send(self, data, loc, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def receive(self, data, loc, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def broadcast(self, data, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def scatter(self, data, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def gather(self, data, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data

    def map(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a serial map of the function over
        the passed data

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if extra_args is not None or extra_kwargs is not None:
            if extra_args is None:
                extra_args=()
            if extra_kwargs is None:
                extra_kwargs={}
            function = lambda a, fn=function, ea=extra_args, ek=extra_kwargs: fn(a, *ea, **ek)

        return list(map(function, data, **kwargs))

    def starmap(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a serial map with unpacking of the function over
        the passed data

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if extra_args is None:
            extra_args=()
        if extra_kwargs is None:
            extra_kwargs={}
        function = lambda a, fn=function, ea=extra_args, ek=extra_kwargs: fn(*a, *ea, **ek)

        return list(map(function, data, **kwargs))

    def apply(self, func, *args, comm=None, main_kwargs=None, **kwargs):
        kwargs['parallelizer'] = self
        if main_kwargs is None:
            main_kwargs = {}
        return func(*args, **main_kwargs, **kwargs)

    def wait(self):
        """
        No need to wait when you're in a serial environment
        :return:
        :rtype:
        """

        pass



