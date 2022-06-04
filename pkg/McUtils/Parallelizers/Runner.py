import typing
from .Parallelizers import Parallelizer

__all__ = [
    "ClientServerRunner"
]

class ClientServerRunner:
    """
    Provides a framework for running MPI-like scripts in a client/server
    model
    """

    def __init__(self, client_runner:typing.Callable, server_runner:typing.Callable, parallelizer:Parallelizer):
        self.client = client_runner
        self.server = server_runner
        self.par = parallelizer

    def run(self):
        """
        Runs the client/server processes depending on if the parallelizer
        is on the main or server processes

        :return:
        :rtype:
        """
        if self.par.on_main:
            self.client()
        else:
            self.server()


