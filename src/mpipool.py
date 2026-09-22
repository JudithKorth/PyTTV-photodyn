from mpi4py import MPI


class MPIPool:

    def __init__(self, comm=None):
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if self.size < 2:
            raise ValueError('MPIPool requires at least two MPI ranks: run with mpiexec -n 2 or more.')

    def is_master(self) -> bool:
        return self.rank == 0

    def wait(self):
        status = MPI.Status()
        while True:
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if task is None:
                break
            func, arg = task
            self.comm.send(func(arg), dest=0, tag=status.Get_tag())

    def map(self, func, tasks):
        tasks = list(tasks)
        ntasks = len(tasks)
        results = [None] * ntasks
        status = MPI.Status()

        inext = 0
        for worker in range(1, min(self.size - 1, ntasks) + 1):
            self.comm.send((func, tasks[inext]), dest=worker, tag=inext)
            inext += 1

        for _ in range(ntasks):
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            results[status.Get_tag()] = result
            if inext < ntasks:
                self.comm.send((func, tasks[inext]), dest=status.Get_source(), tag=inext)
                inext += 1
        return results

    def close(self):
        if self.is_master():
            for worker in range(1, self.size):
                self.comm.send(None, dest=worker, tag=0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
