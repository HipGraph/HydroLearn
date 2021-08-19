from subprocess import Popen
from Arguments import ArgumentBuilder


def invoke(args, n_processes):
    program = "Execute"
    args = args.add_invocation_args(program, prepend=True)
    processes = []
    for i in range(n_processes):
        args = args.replace("process_rank", i, None, ["distributed"])
        processes += [Popen(args.get("args"))]
    for process in processes:
        process.wait()


if __name__ == "__main__":
    from sys import argv
    if len(argv) < 2:
        raise IOError("Expected argument: \"mode\" -> [int: 1 or 2]")
    args = ArgumentBuilder().add_mode_args(argv[1])
    invoke(args, 1)
