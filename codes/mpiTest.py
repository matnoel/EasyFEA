# https://www.youtube.com/watch?v=noMezSAT-w4

# mpiexec -n 2 python3 codes/mpiTest.py

from mpi4py import MPI

import Affichage

Affichage.Clear()

MPI.Init_thread(2)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# if rank == 0:
#     msg = "hello1"
#     comm.send(msg, dest=1)
#     print(f"I am 1 and i send : {msg}")
# elif rank == 1:
#     msg = comm.recv(source=0)
#     print(f"I am 2 and i receive : {msg}")

# deadlock
# essaye de recevoir un message qui n'a pas été envoyé
if rank == 0:
    msg = "hello1"
    comm.send(msg, dest=1)
    print(f"I am 1 and i send : {msg}")
elif rank == 1:
    msg = comm.recv(source=0)
    print(f"I am 2 and i receive : {msg}")