#
# Copyright 2015 Lars Pastewka (U. Freiburg)
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
"""
@file   DistributedClient.py

@author Till Junge <till.junge@kit.edu>

@date   19 Mar 2015

@brief  example for using the  multiprocessing capabilities of PyPyContact ported for matscipy,
        clientside

@section LICENCE

 Copyright (C) 2015 Till Junge

DistributedClient.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

DistributedClient.py is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from matscipy import BaseWorker

class Worker(BaseWorker):
    def __init__(self, address, port, key, worker_id):
        super(Worker, self).__init__(address, port, key.encode())
        self.worker_id = worker_id

    def process(self, job_description, job_id):
        self.result_queue.put((self.worker_id, job_id))

def parse_args():
    parser = Worker.get_arg_parser()
    parser.add_argument('id', metavar='WORKER-ID', type=int, help='Identifier for this process')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    worker = Worker(args.server_address, args.port, args.auth_token, args.id)

    worker.daemon = True
    worker.start()


    while not worker.work_done_flag.is_set():
        worker.job_queue.join()


if __name__ == "__main__":
    main()
