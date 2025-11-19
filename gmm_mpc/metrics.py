import time
from mpi4py import MPI

class Metrics:
    """Enhanced metrics tracking with per-type communication and per-round matching."""

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.start_time = time.time()
        self.rounds = 0
        self.communication_bytes = 0
        self.communication_by_type = {}  # type -> bytes
        self.peak_local_edges = 0
        self.matched_per_round = []  # List of counts per round

    def new_round(self):
        self.rounds += 1

    def record_communication(self, send_counts, dtype_size, msg_type="general"):
        """Record communication with type tracking (Fix B.10)."""
        local_bytes = sum(send_counts) * dtype_size
        total_bytes = self.comm.allreduce(local_bytes, op=MPI.SUM)
        self.communication_bytes += total_bytes
        self.communication_by_type[msg_type] = \
            self.communication_by_type.get(msg_type, 0) + total_bytes

    def update_peak_memory(self, num_edges):
        if num_edges > self.peak_local_edges:
            self.peak_local_edges = num_edges

    def record_matched(self, count):
        """Record number of edges matched in current round (Fix B.10)."""
        self.matched_per_round.append(count)

    def report(self, matching_size):
        """Generate comprehensive metrics report (Fix B.10)."""
        total_time = time.time() - self.start_time

        all_peak_mem = self.comm.gather(self.peak_local_edges, root=0)

        if self.rank == 0:
            print("\n" + "="*60)
            print("METRICS REPORT")
            print("="*60)
            print(f"Total execution time: {total_time:.4f} seconds")
            print(f"Total rounds: {self.rounds}")
            print(f"\nCommunication:")
            print(f"  Total: {self.communication_bytes / 1e6:.4f} MB")
            print(f"  By type:")
            for msg_type, bytes_val in sorted(self.communication_by_type.items()):
                print(f"    {msg_type}: {bytes_val / 1e6:.4f} MB")

            print(f"\nMatching:")
            print(f"  Final matching size: {matching_size} edges")
            if self.matched_per_round:
                print(f"  Matched per round: {self.matched_per_round}")
                print(f"  Total matched across rounds: {sum(self.matched_per_round)}")

            print(f"\nMemory:")
            if all_peak_mem:
                print(f"  Peak edges per rank:")
                print(f"    Max: {max(all_peak_mem)}")
                print(f"    Avg: {sum(all_peak_mem)/len(all_peak_mem):.1f}")
                print(f"    Min: {min(all_peak_mem)}")
            print("="*60)
