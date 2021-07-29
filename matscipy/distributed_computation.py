#
# Copyright 2015 James Kermode (Warwick U.)
#           2015 Till Junge (EPFL)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import multiprocessing
import multiprocessing.managers

try:
    import argparse
except ImportError:
    from matscipy.logger import screen
    screen.pr('argparse module not availability, some functionality disabled')

import abc
import datetime
import sys


class BaseResultManager(object):
    """
    Baseclass for job distribution servers. User needs to implement the method process
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, port, key):
        """

        Keyword Arguments:
        port    -- listening port
        key     -- auth_key
        verbose -- (default False) if set, outputs debugging messages
        """
        self.port = port
        self.key = key
        self.job_queue = None
        self.result_queue = None
        self.todo_counter = None
        self.work_done_flag = None
        self.manager = None
        self.done = False

        self.create_manager()

    def create_manager(self):
        """
        creates a multiprocessing.SyncManager
        """
        self.job_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.JoinableQueue()
        # the -1 is for 'uninitialized
        self.todo_counter = multiprocessing.Manager().Value('i', -1)
        self.work_done_flag = multiprocessing.Manager().Event()
        self.work_done_flag.clear()
        # This is based on the examples in the official docs of multiprocessing.
        # get_{job|result}_q return synchronized proxies for the actual Queue
        # objects.
        class JobQueueManager(multiprocessing.managers.SyncManager):
            pass

        JobQueueManager.register('get_job_queue',
                                 callable=lambda: self.job_queue)
        JobQueueManager.register('get_result_queue',
                                 callable=lambda: self.result_queue)
        JobQueueManager.register('get_todo_counter',
                                 callable=lambda: self.todo_counter,
                                 proxytype= multiprocessing.managers.ValueProxy)
        JobQueueManager.register('get_work_done_event',
                                 callable=lambda: self.work_done_flag,
                                 proxytype= multiprocessing.managers.EventProxy)
        self.manager = JobQueueManager(address=('', self.port), authkey=self.key)
        self.manager.start()

    def set_todo_counter(self, counter):
        self.todo_counter.set(counter)
        self.done = (counter == 0)

    def get_todo_counter(self):
        return self.todo_counter.get()

    def decrement_todo_counter(self):
        new_counter = self.todo_counter.get() - 1
        self.done = (new_counter == 0)
        self.todo_counter.set(self.todo_counter.get() - 1)

    @classmethod
    def get_arg_parser(cls, parser=None):
        """
        create or extend a argparser to read command line arguments required by
        the server.
        Keyword Arguments:
        parser -- optional: if provided, parser is extended to include port and
                  authentication key
        """
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument('--port', type=int, default=9995,
                            help='server listening port')
        parser.add_argument('--auth-token', type=str, default='auth_token',
                            help=('shared information used to authenticate the '
                                  'client to the server'))
        return parser

    def run(self):
        """
        this is the actual serving method. it fills the jobqueue and processes
        incoming results
        """
        print("Start serving jobs and processing results")
        while not self.done:
            self.schedule_available_jobs()
            self.receive_results()
        print()
        print("Signalling end of work to worker processes")
        self.work_done_flag.set()
        print("Waiting for stragglers to hand in results")
        self.result_queue.join()
        print("Wrapping this up")
        self.manager.shutdown()

    @abc.abstractmethod
    def schedule_available_jobs(self):
        """
        to be implemented by inheriting classes. should push available jobs
        into the job queue
        """
        raise NotImplementedError()

    def receive_results(self):
        """
        proposed standard result receiver, can be overloaded by inheriting
        classes
        """
        try:
            result = self.result_queue.get()
            if result:
                value, job_id = result
                self.process(value, job_id)
        finally:
            self.result_queue.task_done()

    @abc.abstractmethod
    def process(self, value, job_id):
        """
        to be implemented by inheriting classes. should push available jobs
        into the job queue
        """
        raise NotImplementedError()


class BaseWorker(multiprocessing.Process):
    """
    Baseclass for distributed calculation worker threads
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, server_address, port, key, verbose=False, walltime=None):
        """

        Keyword Arguments:
        server_address -- ip or fully qualified hostname
        port           -- listening port
        key            -- auth_key
        verbose        -- (default False) if set, outputs debugging messages
        walltime       -- (default None) if set, worker commits suicide after
                          walltime hours
        """
        super(BaseWorker, self).__init__()
        self.server_address = server_address
        self.port = port
        self.key = key

        self.job_queue = None
        self.result_queue = None
        self.todo_counter = None
        self.work_done_flag = None
        self.manager = None

        self.create_manager()
        self.verbose = verbose
        self.commit_suicide = walltime is not None
        self.time_of_death = None
        if self.commit_suicide:
            self.time_of_death = (datetime.datetime.now() +
                                  datetime.timedelta(hours = walltime))


    def create_manager(self):
        """
        creates a multiprocessing.SyncManager
        """
        self.job_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.JoinableQueue()
        # the -1 is for 'uninitialized
        self.todo_counter = multiprocessing.Manager().Value('i', -1)
        self.work_done_flag = multiprocessing.Manager().Event()
        self.work_done_flag.clear()

        # This is based on the examples in the official docs of multiprocessing.
        # get_{job|result}_q return synchronized proxies for the actual Queue
        # objects.
        class ServerQueueManager(multiprocessing.managers.SyncManager):
            pass

        ServerQueueManager.register('get_job_queue')
        ServerQueueManager.register('get_result_queue')
        ServerQueueManager.register('get_todo_counter')
        ServerQueueManager.register('get_work_done_event')

        self.manager = ServerQueueManager(
            address=(self.server_address, self.port),
             authkey=self.key)
        self.manager.connect()

        self.job_queue = self.manager.get_job_queue()
        self.result_queue = self.manager.get_result_queue()
        self.todo_counter = self.manager.get_todo_counter()
        self.work_done_flag = self.manager.get_work_done_event()
        return self.manager

    @classmethod
    def get_arg_parser(cls, parser=None):
        """
        create or extend a argparser to read command line arguments required by
        the cliend.
        Keyword Arguments:
        parser -- optional: if provided, parser is extended to include port and
                  authentication key
        """
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument('--server_address', metavar='INET_ADDR', type=str,
                            default='',
                            help=('job server ip address or fully qualified '
                                  'hostname'))
        parser.add_argument('--port', type=int, default=9995,
                            help='server listening port')
        parser.add_argument('--auth-token', type=str, default='auth_token',
                            help=('shared information used to authenticate the '
                                  'client to the server'))
        return parser


    def run(self):
        """
        standard method that any multiprocessing.Process must implement
        """
        if self.verbose:
            print("Starting to run")

        if self.commit_suicide:
            def gotta_commit_suicide():
                do_I = datetime.datetime.now() > self.time_of_death
                if do_I:
                    print("Reached walltime, stopping accepting new jobs (zombie)")
                return do_I
        else:
            def gotta_commit_suicide():
                return False

        while not self.work_done_flag.is_set() and not gotta_commit_suicide():
            try:
                if self.verbose:
                    print("trying to get a job")
                job_description, job_id = self.job_queue.get()
                if self.verbose:
                    print("got job {}".format(job_id))
                try:
                    self.process(job_description, job_id)
                except Exception as err:
                    print("ERROR:::: {}".format(err))
                    raise
            finally:
                try:
                    self.job_queue.task_done()
                except EOFError:
                    pass

    @abc.abstractmethod
    def process(self, job_description, job_id):
        raise NotImplementedError()
