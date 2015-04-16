import logging
import os
import shutil
import subprocess
import socket
import SocketServer
import StringIO
import time
import threading
from Queue import Queue

import numpy as np

from matscipy.elasticity import full_3x3_to_Voigt_6_stress

from ase.atoms import Atoms
from ase.io.extxyz import read_xyz, write_xyz
from ase.io.vasp import write_vasp
from ase.calculators.calculator import Calculator
from ase.calculators.vasp import Vasp

MSG_LEN_SIZE = 8
MSG_END_MARKER = 'done.\n'
MSG_END_MARKER_SIZE = len(MSG_END_MARKER)
MSG_INT_SIZE = 6
MSG_FLOAT_SIZE = 25
MSG_FLOAT_FORMAT = '%25.16f'
MSG_INT_FORMAT = '%6d'

ATOMS_REQUESTS = {'A': 'REFTRAJ', 'X': 'XYZ'}
RESULTS_REQUESTS = {'R': 'REFTRAJ', 'Y': 'XYZ'}
ZERO_ATOMS_DATA = {'REFTRAJ': '     242     0\n     0\n       0.0000000000000000       0.0000000000000000       0.0000000000000000\n       0.0000000000000000       0.0000000000000000       0.0000000000000000\n       0.0000000000000000       0.0000000000000000       0.0000000000000000\n',
                   'XYZ': '     2500\nlabel=0 cutoff_factor=1.20000000 nneightol=1.20000000 Lattice="0.00000000       0.00000000       0.00000000       0.00000000       0.00000000       0.00000000       0.00000000       0.00000000       0.00000000" Properties=species:S:1:pos:R:3:Z:I:1\n'}
CLIENT_TIMEOUT = 60
MAX_RMS_DIFF = 1.0   # angstrom 
MAX_CELL_DIFF = 1e-3 # angstrom

def pack_atoms_to_reftraj_str(at, label):
    data = ''
    data += MSG_INT_FORMAT % label + '\n'
    data += MSG_INT_FORMAT % len(at) + '\n'
    for i in range(3):
        data += (3*MSG_FLOAT_FORMAT) % tuple(at.cell[:, i]) + '\n'
    g = np.linalg.inv(at.cell.T).T
    for i in range(len(at)):
        data += (3*MSG_FLOAT_FORMAT) % tuple(np.dot(g, at.positions[i, :])) + '\n'

    # preceed message by its length
    data_length = ('%8d' % len(data)).encode('ascii')
    data = data_length + data.encode('ascii')
    return data

def pack_atoms_to_xyz_str(at, label):
    at.info['label'] = label
    buffer = StringIO.StringIO()
    write_xyz(buffer, at)
    data = str(buffer)
    buffer.close()    
    # preceed message by its length
    data_length = ('%8d' % len(data)).encode('ascii')
    data = data_length + data.encode('ascii')
    return data

def unpack_reftraj_str_to_atoms(data):
    lines = data.split('\n')
    label = int(lines[0])
    n_atoms = int(lines[1])
    at = Atoms(symbols=[' ']*n_atoms, cell=np.eye(3))
    at.info['label'] = label
    for i in range(3):
        at.cell[:, i] = [float(x) for x in lines[i].split()]
    for i, line in enumerate(lines[4:]):
        t = [float(x) for x in line.split()]
        at.positions[i, :] = np.dot(at.cell.T, t)
    return at

def pack_results_to_reftraj_output_str(at):
    data = ''
    data += MSG_INT_FORMAT % len(at) + '\n'
    data += MSG_FLOAT_FORMAT % at.energy + '\n'
    force = at.get_array('force')
    virial = at.info['virial']
    for i in at.indices:
        data += (3*MSG_FLOAT_FORMAT) % tuple(force[i, :]) + '\n'
    # NB: not in Voigt order (xx, yy, zz, yz, xz, xy)
    data += (6*MSG_FLOAT_FORMAT) % (virial[0,0], virial[1,1], virial[2,2],
                                    virial[0,1], virial[1,2], virial[0,2]) 

    # preceed message by its length
    data_length = ('%8s' % len(data)).encode('ascii')
    data = data_length + data
    return data

def unpack_reftraj_output_str_to_results(data):
    lines = data.strip().split('\n')
    label = int(lines[0])
    natoms = int(lines[1])
    energy = float(lines[2])
    force = np.zeros((natoms,3))
    for i, line in enumerate(lines[3:-1]):
       force[i, :] = [float(f) for f in line.split()]
    v6 = [float(v) for v in lines[-1].split()]
    virial = np.zeros((3,3))
    # NB: not in Voigt order (xx, yy, zz, yz, xz, xy)    
    virial[0,0], virial[1,1], virial[2,2], virial[0,1], virial[1,2], virial[0,2] = v6
    virial[1,0] = virial[0,1]
    virial[2,1] = virial[1,2]
    virial[2,0] = virial[0,2]
    return (label, (natoms, energy, force, virial))

def unpack_xyz_str_to_results(data):
    buffer = StringIO.StringIO(data)
    at = read_xyz(buffer)
    buffer.close()
    label = at.info['label']
    return (label, at)

class AtomsRequestHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        ip, port = self.client_address
        task = None
        # receive request code and client ID
        request_str = self.rfile.read(MSG_LEN_SIZE)
        request = request_str[0]
        client_id = int(request_str[1:])
        
        if client_id > self.server.njobs-1:
            raise RuntimeError('Unknown client ID %d outside of range 0 < ID < %d' % 
                               (client_id, self.server.njobs-1))

        logging.info('"%s" request from %s:%d client %d' % (request, ip, port, client_id))
        #print 'input queue lengths ', ''.join(['%d:%d ' % (i,q.qsize()) for (i,q) in enumerate(input_qs)])
        #print 'output queue length %d' % output_q.qsize()

        if request in ATOMS_REQUESTS:
            # client is ready for Atoms (in either REFTRAJ or XYZ format)
            data, fmt, label, at = self.server.input_qs[client_id].get()
            assert ATOMS_REQUESTS[request] == fmt
            if data == 'shutdown' or data == 'restart':
                task = data
                data = ZERO_ATOMS_DATA[fmt]
            self.wfile.write(data)

        elif request in RESULTS_REQUESTS:
            # results are available from client in REFTRAJ or XYZ format
            data_size = int(self.rfile.read(MSG_LEN_SIZE))
            data = self.rfile.read(data_size)
            fmt = RESULTS_REQUESTS[request]
            self.server.output_q.put((client_id, fmt, data))
            self.server.input_qs[client_id].task_done()

        else:
            raise RuntimeError('Unknown request code "%s"' % request)

        # say goodbye to this client
        self.wfile.write(MSG_END_MARKER)

        if (request == 'A' or request == 'X') and task == 'restart':
            # if we're restarting a client, get the next thing out of the queue
            # and re-initialise. Restart won't do anything until shutdown
            # of old client has completed.
            data, fmt, label, at = self.server.input_qs[client_id].get()
            logging.info('"%s" request from client %d triggering restart for calculation with label %d' % 
                         (request, client_id, label))
            self.server.clients[client_id].start_or_restart(at, label, restart=True)
        

class AtomsServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, clients,
                 bind_and_activate=True, max_attempts=3, bgq=False):
        
        self.njobs = len(clients)
        # allow up to twice as many threads as sub-block jobs
        self.request_queue_size = 2*self.njobs
        self.max_attempts = max_attempts
        self.bgq = bgq # If True, we're running on IBM Blue Gene/Q platform

        SocketServer.TCPServer.__init__(self, 
                                        server_address, 
                                        RequestHandlerClass,
                                        bind_and_activate)

        self.clients = clients
        for client in self.clients:
            client.server = self # FIXME circular reference

        # record all input in the order in which it is put()
        self.input_q = Queue()
        # we also need an input Queue for each client: this is so that we can
        # exploit wavefunction reuse by sending consecutive clusters belonging
        # to the same atom to the same QM partition
        self.input_qs = [Queue() for i in range(self.njobs) ]
        self.output_q = Queue()



    def server_activate(self):
        SocketServer.TCPServer.server_activate(self)
        self.ip, self.port = self.server_address
        if self.bgq:
            # If we're on a Blue Gene, note that IP address returned
            # by server.server_address is not the correct one for CNs
            # to talk to FEN, so we discard it, and use the InfiniBand
            # address returned by get_hostname_ip()
            import bgqtools
            hostname, self.ip = bgqtools.get_hostname_ip()
        else:
            hostname = socket.gethostname()
        logging.info('AtomsServer running on %s %s:%d with njobs=%d' % 
                     (hostname, self.ip, self.port, self.njobs))


    def shutdown_clients(self):
        logging.info('shutting down all clients')
        wait_threads = []
        for client_id, client in enumerate(self.clients):
            if (client.process is not None and client.process.poll() is None and 
                (client.wait_thread is None or not client.wait_thread.isAlive())):
                wait_threads.append(client.shutdown(block=False))
        # wait for them all to finish shutting down
        for wait_thread in wait_threads:
            if wait_thread is None or not wait_thread.isAlive():
                continue
            wait_thread.join()
        logging.info('all client shutdowns complete')


    def shutdown(self):
        self.shutdown_clients()
        return SocketServer.TCPServer.shutdown(self)


    def put(self, at, client_id, label, force_restart=False):
        logging.info('Putting Atoms to client %d label %d' % (client_id, label))

        # allow client to modify atoms (e.g. sort them)
        at, fmt, first_time = self.clients[client_id].preprocess(at, label, force_restart)

        # store what we actually did- `at` may have been modified by preprocess()
        self.input_q.put((label, client_id, at))

        if fmt == 'REFTRAJ':
            data = pack_atoms_to_reftraj_str(at, label)
        elif fmt == 'XYZ':
            data = pack_atoms_to_xyz_str(at, label)
        else:
            raise ValueError('Unknown format "%s"' % fmt)
        self.input_qs[client_id].put((data, fmt, label, at))

        if first_time:
            # throw away what we just put(), as it's in the input files.
            # note that we don't call task_done() until results come in
            discard = self.input_qs[client_id].get()


    def join_all(self):
        logging.info('AtomsServer waiting for input queues to empty')
        for input_q in self.input_qs:
            input_q.join()
        logging.info('all AtomsServer queues drained.')


    def get_results(self):
        logging.info('AtomsServer getting results')

        results = {}

        for attempt in range(self.max_attempts):
            rejects = []
            self.join_all()
            logging.info('AtomsServer.get_results() attempt %d of %d jobs finished' %
                         (attempt+1, self.max_attempts))

            while self.output_q.unfinished_tasks:
                client_id, fmt, data = self.output_q.get()

                if fmt == 'REFTRAJ':
                    label, res = unpack_reftraj_output_str_to_results(data)
                elif fmt == 'XYZ':
                    label, res = unpack_xyz_str_to_results(data)
                else:
                    raise ValueError('get_results() got unknown format "%s"' % fmt)

                if label > 0: # WARNING: labels must start from 1, or first calc never passes test
                    # calculation converged, save the results
                    logging.info('calculation label %d client %d CONVERGED' % (label, client_id))
                    results[label] = res
                else:
                    # calculation did not converge, we need to repeat it
                    logging.info('calculation label %d client %d DID NOT CONVERGE' % (label, client_id))
                    rejects.append(-label)

                self.output_q.task_done()

            logging.debug('AtomsServer.get_results() rejects=%r' % rejects)
            logging.debug('AtomsServer.get_results() sorted(results.keys())=%r' % sorted(results.keys()))

            # collect all input task so we can see if anything is missing
            input = {}
            while self.input_q.unfinished_tasks:
                label, client_id, at = self.input_q.get()
                input[label] = (client_id, at)
                self.input_q.task_done()

            logging.debug('AtomsServer.get_results() sorted(input.keys())=%r' % sorted(input.keys()))

            # resubmit any failed calculations
            for label in rejects:
                client_id, at = input[label]
                logging.info('Resubmiting calculation label %d client_id %d' % (label, client_id))
                self.put(at, client_id, label, force_restart=True)

            assert len(results) + len(rejects) == len(input)

            # if all calculations converged we are done
            if len(rejects) == 0:
                break
        else:
            raise RuntimeError('max_attempts (%d) exceeded without all calculations completing succeasfully' %
                               self.max_attempts)

        assert(len(results) == len(input))
        assert(len(rejects) == 0)

        results_atoms = []
        for (inp_label, label) in zip(sorted(input.keys()), sorted(results.keys())):

            assert inp_label == label
            client_id, inp_at = input[inp_label]
            res = results[label]

            if isinstance(res, Atoms):
                at = res
            else:
                (natoms, energy, force, virial) = res
                assert len(inp_at) == natoms
                
                at = inp_at.copy() # FIXME could possibly store results inplace, but need to think about sorting
                at.info['label'] = label
                at.info['energy'] = energy
                at.set_array('force', force)
                at.info['virial'] = virial

            # allow client to modify results (e.g. reverse sort order)
            at = self.clients[client_id].postprocess(at, label)

            results_atoms.append(at)

        logging.info('AtomsServer processed %d results' % len(results))
        return results_atoms


class Client(object):
    """
    Represents a single Client job

    Used by AtomsServer to start, restart and shutdown clients
    running on the Compute Nodes.
    """

    def __init__(self, client_id, exe, env=None, npj=1, ppn=1,
                 block=None, corner=None, shape=None,
                 jobname='socketcalc', rundir=None,
                 fmt='REFTRAJ', parmode=None, mpirun='mpirun'):

        self.client_id = client_id
        self.process = None # handle for the runjob process
        self.log = None # stdout file
        self.wait_thread = None # used by shutdown(block=False)
        self.last_atoms = None # used to check if we can continue from previous task
        self.lock = threading.Lock() # avoid concurrancy issues

        if env is None:
            env = {}
        self.env = env # environment
        self.exe = exe # executable 
        self.npj = npj # nodes per job
        self.ppn = ppn # processes per node

        self.block, self.corner, self.shape = block, corner, shape

        self.jobname = jobname
        self.fmt = fmt
        self.parmode = parmode
        self.mpirun = mpirun

        self.rundir = rundir or os.getcwd()
        self.subdir = os.path.join(self.rundir, '%s-%03d' % (jobname, self.client_id))
        if not os.path.exists(self.subdir):
            logging.info('Making subdir %s' % self.subdir)
            os.mkdir(self.subdir)

    def extra_args(self, label=None):
        """
        Return list of additional command line arguments to be passed to client
        """
        args = [self.server.ip, str(self.server.port), str(self.client_id)]
        if label is not None:
            args.append(str(label))
        return args

    def start(self, label=None):
        """
        Start an individual client.

        Raises RuntimeError if this client is already running.
        """
        if self.process is not None:
            raise RuntimeError('client %d is already running' % client_id)

        runjob_args = []
        popen_args = {}
        if self.parmode == 'cobalt':
            # Convert env to "--envs KEY=value" arguments for runjob
            envargs = []
            for (k, v) in self.env.iteritems():
                envargs.extend(['--envs', '%s=%s' % (k, v) ])
            
            runjob_args += ['runjob', '--block', self.block]
            if self.corner is not None:
                runjob_args += ['--corner', self.corner]
            if self.shape is not None:
                runjob_args += ['--shape', self.shape]
            runjob_args += (['-n', str(self.npj*self.ppn), '-p', str(self.ppn)] + envargs + 
                            ['--cwd', self.subdir, ':'])
        elif self.parmode == 'mpi':
            runjob_args += [self.mpirun, '-np', str(self.npj*self.ppn)]
            popen_args['cwd'] = self.subdir
            popen_args['env'] = os.environ # for mpi, let mpirun inherit environment of script
        else:
            popen_args['cwd'] = self.subdir
            popen_args['env'] = self.env
        runjob_args += [self.exe]
        runjob_args += self.extra_args(label)
        logging.info('starting client %d args %r' % (self.client_id, runjob_args))
        self.log = open(os.path.join(self.rundir, '%s-%03d.output' % (self.jobname, self.client_id)), 'a')
        # send stdout and stderr to same file
        self.process = subprocess.Popen(runjob_args, stdout=self.log, stderr=self.log, **popen_args) 


    def shutdown(self, block=True):
        """Request a client to shutdown.

        If block=True, does not return until shutdown is complete.  If
        block=False, waits for the client to shutdown in a new
        thread. Check self.waits_thread.isAlive() to see when shutdown
        has finished. (This function also returns a handle to the wait
        thread when block=False).
        """
        if self.process is None:
            loggin.warn('client %d (requested to shutdown) has never been started' % self.client_id)
            return

        if self.process.poll() is not None:
            logging.warn('client %d is already shutdown' % self.client_id)
            return

        if (self.wait_thread is not None and self.wait_thread.isAlive()):
            raise RuntimeError('client %d is already in the process of shutting down' % self.client_id)

        input_q = self.server.input_qs[self.client_id]
        input_q.put(('shutdown', self.fmt, -1, None))

        if block:
            self.wait_for_shutdown()
        else:
            self.wait_thread = threading.Thread(target=self.wait_for_shutdown)
            self.wait_thread.start()
            return self.wait_thread
            

    def wait_for_shutdown(self):
        """
        Block until a client has shutdown.

        Typically called automatically by shutdown() or
        start_or_restart().

        Shutdown should previously have been initiated by queuing a
        'shutdown' or 'restart' request. Waits CLIENT_TIMEOUT for
        graceful shutdown. If client is still alive, a SIGTERM signal
        is sent. If this has had no effect after a further
        CLIENT_TIMEOUT, then a SIGKILL is sent. Does not return until
        the SIGKILL has taken effect.
        
        This function also marks shutdown task as complete in
        servers's input_q for this client.
        """
        wait_thread = threading.Thread(target=self.process.wait)
        logging.info('waiting for client %d to shutdown' % self.client_id)
        wait_thread.start()
        wait_thread.join(CLIENT_TIMEOUT)
        if wait_thread.isAlive():
            logging.info('client %d did not shutdown gracefully in %d seconds - sending SIGTERM' %
                         (self.client_id, CLIENT_TIMEOUT))
            self.process.terminate()
            wait_thread.join(CLIENT_TIMEOUT)
            if wait_thread.isAlive():
                logging.info('client %d did not respond to SIGTERM - sending SIGKILL' % self.client_id)
                self.process.kill()
                wait_thread.join() # no timeout for kill
            else:
                logging.info('client %d responded to SIGTERM' % self.client_id)
        else:
            logging.info('client %d shutdown within timeout' % self.client_id)
        logging.info('client %d shutdown complete - exit code %r' % (self.client_id, self.process.poll()))
        self.log.close()

        self.process = None
        self.log = None
        self.server.input_qs[self.client_id].task_done()
                                               

    def start_or_restart(self, at, label, restart=False):
        """
        Start or restart a client

        If restart=True, wait for previous client to shutdown first.
        Calls write_input_files() followed by start().
        """
        if restart:
            self.wait_for_shutdown()
        self.write_input_files(at, label)
        self.start(label)


    def preprocess(self, at, label, force_restart=False):
        """
        Prepare client for a calculation.

        Starts client if this is the first task for it, or schedules a
        restart if new configuration is not compatible with the last
        one submitted to the queue (see is_compatible() method).

        Many be extended in subclasses to e.g. sort the atoms by
        atomic numbe. If Atoms object needs to be changed, a copy
        should be returned rather than updating it inplace.

        Returns (at, first_time). 
        """

        first_time = self.process is None
        restart_reqd = (not first_time and (force_restart or 
                                            (not self.is_compatible(self.last_atoms, at, label))))

        # keep a copy of last config queued for this client.
        # acquire a lock in case multiple put() calls to the same client
        # occur concurrently.
        try:
            self.lock.acquire()
            self.last_atoms = at.copy()
        finally:
            self.lock.release()

        if restart_reqd:
            # put a shutdown command into the queue, ahead of this config.
            # once it gets completed, restart_client() will be called as below
            logging.warn('restart scheduled for client %d label %d' % (self.client_id, label))
            self.server.input_qs[self.client_id].put(('restart', self.fmt, -1, None))
        if first_time:
            self.start_or_restart(at, label, restart=False)

        return at, self.fmt, first_time


    def postprocess(self, at, label):
        """
        Post-process results of calculation.

        May be overrriden in subclasses to e.g. reverse sort order
        applied in preprocess().
        """
        return at

    
    def is_compatible(self, old_at, new_at, label):
        """
        Check if new_at and old_at are compatible.
        
        Returns True if calculation can be continued, or False
        if client must be restarted before it can process new_at.
        """
        if old_at is None:
            return True

        return True


    def write_input_files(self, at, label):
        raise NotImplementedError('to be implemented in subclasses')


        
class QUIPClient(Client):
    """
    Subclass of Client for running QUIP calculations.

    Initial input files are written in extended XYZ format, and
    subsequent communication is via sockets, in either REFTRAJ
    or XYZ format.
    """

    def __init__(self, client_id, exe, env=None, npj=1, ppn=1,
                 block=None, corner=None, shape=None,
                 jobname='socketcalc', rundir=None,
                 fmt='REFTRAJ', parmode=None, mpirun='mpirun', param_files=None):
        Client.__init__(self, client_id, exe, env, npj, ppn,
                        block, corner, shape, jobname, rundir, fmt, parmode, mpirun)
        self.param_files = param_files

    def write_input_files(self, at, label):
        write_xyz(os.path.join(self.subdir, 'atoms.%d.xyz' % self.client_id), at)

        # copy in parameter files
        if self.param_files is not None:
            for param_file in self.param_files:
                param_file_basename = os.path.basename(param_file)
                shutil.copyfile(param_file, os.path.join(self.subdir, param_file_basename))


_chdir_lock = threading.Lock()

class VaspClient(Client):
    """
    Subclass of Client for running VASP calculations.

    Initial input files are written in POSCAR, INCAR, POTCAR and KPOINTS
    formats, and subsequent communicatin is via sockets in REFTRAJ format.
    """

    def __init__(self, client_id, exe, env=None, npj=1, ppn=1,
                 block=None, corner=None, shape=None,
                 jobname='socketcalc', rundir=None,
                 fmt='REFTRAJ', parmode=None, mpirun='mpirun', **vasp_args):
        Client.__init__(self, client_id, exe, env, npj, ppn,
                        block, corner, shape, jobname, rundir, fmt, parmode, mpirun)
        if 'ibrion' not in vasp_args:
            logging.warn('No ibrion key in vasp_args, setting ibrion=13')
            vasp_args['ibrion'] = 13
        self.vasp_args = vasp_args

    def is_compatible(self, old_at, new_at, label):
        # first time, anything goes
        if old_at is None:
            return True

        if not Client.is_compatible(self, old_at, new_at, label):
            return False

        if len(old_at) != len(new_at):
            logging.info('is_compatible() on client %d label %d got number of atoms mismatch: %d != %d' % (self.client_id,
                                                                                                           label,
                                                                                                           len(old_at),
                                                                                                           len(new_at)))
            return False # number of atoms must match

        if abs(old_at.cell - new_at.cell).max() > MAX_CELL_DIFF:
            logging.info('is_compatible() on client %d label %d got cell mismatch: %r != %r' % (self.client_id,
                                                                                                   label,
                                                                                                   old_at.cell,
                                                                                                   new_at.cell))
            return False # cells must match
        
        # RMS difference in positions must be less than MAX_RMS_DIFF
        old_p = old_at.get_positions()
        new_p = new_at.get_positions()
        
        old_z = old_at.get_chemical_symbols()
        new_z = new_at.get_chemical_symbols()

        if 'index' in old_at.arrays:
            old_index = old_at.get_array('index')
            new_index = new_at.get_array('index')
            
            # if termination exists, undo ordering differences due to cluster hopping
            if ('termindex_%d' % self.client_id) in old_at.arrays:
                old_termindex = old_at.get_array('termindex_%d' % self.client_id)
                new_termindex = new_at.get_array('termindex_%d' % self.client_id)
                
                a1s = sorted([(old_index[i], old_z[i], list(old_p[i]))
                              for i in range(len(old_at)) if old_termindex == 0])

                a2s = sorted([(new_index[i], new_z[i], list(new_p[i])) 
                              for i in range(len(new_at)) if new_termindex == 0])
            else:
                a1s = sorted([(old_index[i], old_z[i], list(old_p[i])) for i in range(len(old_at)) ])
                a2s = sorted([(new_index[i], new_z[i], list(new_p[i])) for i in range(len(new_at)) ])

            old_p = np.r_[[p for (i, z, p) in a1s]]
            new_p = np.r_[[p for (i, z, p) in a2s]]

            old_z = np.r_[[z for (i, z, p) in a1s]]
            new_z = np.r_[[z for (i, z, p) in a2s]]

        if not np.all(old_z == new_z):
            logging.info('is_compatible() on client %d label %d got atomic number mismatch: %r != %r' % (self.client_id,
                                                                                                         label,
                                                                                                         old_z, new_z))
            return False # atomic numbers must match

        # undo jumps across PBC - approach is that of QUIP's undo_pbc_jumps() routine
        old_g = np.linalg.inv(old_at.cell.T).T
        d = new_p.T - old_p.T - (np.dot(old_at.cell, np.floor(np.dot(old_g, (new_p - old_p).T)+0.5)))
        rms_diff = np.sqrt((d**2).mean())
        logging.info('is_compatible() on client %d label %d got RMS position difference %.3f' % (self.client_id, label, rms_diff))

        if rms_diff > MAX_RMS_DIFF:
            logging.info('is_compatible() on client %d label %d got RMS position difference %.3f > MAX_RMS_DIFF=%.3f' %
                         (self.client_id, label, rms_diff, MAX_RMS_DIFF))
            return False

        return True


    def preprocess(self, at, label, force_restart=False):
        logging.info('vasp client %d preprocessing atoms label %d' % (self.client_id, label))
        # call the parent method first
        at, fmt, first_time = Client.preprocess(self, at, label, force_restart)

        # make a copy and then sort atoms in the same way that vasp
        # calculator will when it writes POSCAR. We use a new
        # calculator and store the sort order in the Atoms so it can
        # be reversed when results are ready.
        vasp = Vasp(**self.vasp_args)
        vasp.initialize(at)
        at = at.copy()
        order = np.array(range(len(at)))
        at.set_array('vasp_sort_order', order)
        at = at[vasp.resort]

        return at, fmt, first_time


    def postprocess(self, at, label):
        logging.info('vasp client %d postprocessing atoms label %d' % (self.client_id, label))
        # call the parent method first
        at = Client.postprocess(self, at, label)
        # restore original atom ordering
        at = at[at.arrays['vasp_sort_order']]
        return at


    def write_input_files(self, at, label):
        global _chdir_lock
        vasp = Vasp(**self.vasp_args)
        vasp.initialize(at) 
        # chdir not thread safe, so acquire global lock before using it
        orig_dir = os.getcwd()
        try:
            _chdir_lock.acquire()
            os.chdir(self.subdir)
            if os.path.exists('OUTCAR'):
                n = 1
                while os.path.exists('OUTCAR.%d' % n):
                    n += 1
                shutil.copyfile('OUTCAR', 'OUTCAR.%d' % n)
                shutil.copyfile('POSCAR', 'POSCAR.%d' % n)
            write_vasp('POSCAR', vasp.atoms_sorted,
                       symbol_count=vasp.symbol_count,
                       vasp5='5' in self.exe)
            vasp.write_incar(at)
            vasp.write_potcar()
            vasp.write_kpoints()
        finally:
            os.chdir(orig_dir)
            _chdir_lock.release()


class SocketCalculator(Calculator):
    """
    ASE-compatible calculator which communicates with remote 
    force engines via sockets using an AtomsServer.
    """

    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, client, ip=None, atoms=None, port=0):
        self.client = client
        if ip is None:
            ip = '127.0.0.1' # default to localhost
        self.server = AtomsServer((ip, port), AtomsRequestHandler, [self.client])
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.label = 1
        self.atoms = atoms

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes: # if anything at all changed (could be made more fine-grained)
            self.server.put(atoms, 0, self.label)
            self.label += 1
            [results] = self.server.get_results()
            self.results['energy'] = results.info['energy']
            self.results['forces'] = results.arrays['force']
            stress = -(results.info['virial']/results.get_volume())
            self.results['stress'] = full_3x3_to_Voigt_6_stress(stress)


    def shutdown(self):
        self.server.shutdown()
                        
