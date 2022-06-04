import os
import sys
import signal
from subprocess import Popen
from importlib import import_module
import Utility as util
from Container import Container
from Arguments import ArgumentParser, ArgumentBuilder, arg_flag


def signal_handler(sig, frame):
    print("Exiting from interrupt Ctrl+C")
    sys.exit()


class Executor:

    debug = False
    
    def __init__(self, interpreter="python", program="Execute.py", n_processes=1):
        self.interpreter, self.program, self.n_processes = interpreter, program, n_processes
        signal.signal(signal.SIGINT, self.on_interrupt)

    def invoke(self, jobs):
        arg_bldr, chkptr = ArgumentBuilder(), Checkpointer()
        i = -1
        while i < len(jobs) - 1:
            i += 1
            job = jobs[i]
            if job.chkpt:
                if chkptr.is_completed(job):
                    print("JOB ALREADY COMPLETED. MOVING ON...")
                    continue
            args = Container().copy(job.work)
            if not args.is_empty() and False:
                print(util.make_msg_block("Arguments %2d" % (i+1)))
                print(arg_bldr.view(args))
            if self.debug:
                print(util.make_msg_block("Arguments %2d" % (i+1)))
                print(job.work)
            if self.debug:
                continue
            processes = []
            for j in range(self.n_processes):
                args.set("process_rank", j, None, ["distribution"])
                invocation = [self.interpreter, self.program] + arg_bldr.build(args)
                processes += [Popen(invocation)]
            self.interrupted, exited_correctly = False, True
            for process in processes:
                returncode = process.wait()
                exited_correctly = exited_correctly and not returncode
            if self.interrupted or not exited_correctly:
                print("Job was interrupted or exited incorrectly and will not be checkpointed")
                check = input("Repeat, next, or quit? [r,n,q]: ")
                if "r" in check.lower():
                    i -= 1
                elif "q" in check.lower():
                    break
                elif "n" in check.lower():
                    pass
                else:
                    raise ValueError("Unknown option \"%s\". Quitting..." % (check))
            elif job.chkpt:
                chkptr.checkpoint(job)

    def on_interrupt(self, sig, frame):
        self.interrupted = True


class Checkpointer:

    def __init__(self):
        pass

    def get_checkpoint_contents(self, job):
        return ArgumentBuilder().view(job.work)

    def checkpoint(self, job):
        path = job.checkpoint_path()
        chkpt_dir = job.checkpoint_dir()
        os.makedirs(chkpt_dir, exist_ok=True)
        contents = self.get_checkpoint_contents(job)
        with open(path, "a+") as f:
            f.write(contents + "\n\n")

    def is_completed(self, job):
        path = job.checkpoint_path()
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            completed_jobs = f.read()
        contents = self.get_checkpoint_contents(job)
        return contents in completed_jobs


class Job:

    def __init__(self, work, chkpt=False):
        if isinstance(work, list):
            work = ArgumentParser().parse_arguments(work)
        elif not isinstance(work, Container):
            raise ValueError("Parameter \"work\" must be a list (sys.argv) or Container. Received %s" % (
                str(type(work))
            ))
        self.name = "Job"
        self.work = work
        self.chkpt = chkpt

    def root_dir(self):
        return os.sep.join(__file__.split(os.sep)[:-1])

    def checkpoint_dir(self):
        return os.sep.join([self.root_dir(), "Cache"])

    def checkpoint_path(self):
        return os.sep.join([self.checkpoint_dir(), "CompletedJobs.txt"])

    def __str__(self):
        return self.work.to_string(extent=[-1, -1])


class Driver:

    interpreter = "python"
    program = "Execute.py"
    n_processes = 1
    debug = False

    def __init__(self, args):
        driver_args = None
        if "driver" in args:
            driver_args = args.driver
            args.rem("driver")
        if not driver_args is None:
            self.debug = driver_args.debug if "debug" in driver_args else False
            if "interpreter" in driver_args: self.interpreter = driver_args.interpreter
            if "program" in driver_args: self.program = driver_args.program
            if "n_processes" in driver_args: self.n_processes = driver_args.n_processes

    def run(self, args):
        if "E" in args: # Invoking a set of jobs defined by an experiment
            if not "e" in args:
                raise ValueError("Given %sE but missing %se" % (arg_flag, arg_flag))
            exp_module_name, exp_id = args.get(["E", "e"])
            args.rem(["E", "e"])
            exp_module = import_module("Experimentation.%s" % (exp_module_name))
            if isinstance(exp_id, list): # Multiple experiments: collect jobs from all
                jobs = []
                for _exp_id in exp_id:
                    exp = getattr(exp_module, "Experiment%s" % (str(_exp_id)))
                    jobs += exp().jobs
            else:
                exp = getattr(exp_module, "Experiment%s" % (str(exp_id)))
                jobs = exp().jobs
        elif "A" in args: # Invoking an anlysis
            if not "a" in args:
                raise ValueError("Given %sE but missing %se" % (arg_flag, arg_flag))
            analysis_module_name, analysis_id = args.get(["A", "a"])
            args.rem(["A", "a"])
            analysis_module = import_module("Analysis.%s" % (analysis_module_name))
            if not isinstance(analysis_id, list):
                analysis_id = [analysis_id]
            for _analysis_id in analysis_id:
                analysis = getattr(analysis_module, "Analysis%s" % (str(_analysis_id)))
                analysis().run(args)
            sys.exit()
        elif "I" in args: # Invoking data integration
            integration_module = import_module("Data.Integration")
            integration_class_name = args.get("I")
            args.rem("I")
            acquire, convert = True, True
            if "acquire" in args:
                acquire = args.get("acquire")
                args.rem("acquire")
            if "convert" in args:
                convert = args.get("convert")
                args.rem("convert")
            if not isinstance(integration_class_name, list):
                integration_class_name = [integration_class_name]
            for _integration_class_name in integration_class_name:
                integration_class = getattr(integration_module, _integration_class_name)
                integration = integration_class()
            if acquire:
                integration.acquire(args)
            if convert:
                integration.convert(args)
            sys.exit()
        else:
            jobs = [Job(args)]
        for job in jobs: # Add driver-level args to all jobs
            job.work.copy(args)
        jobs = jobs[:]
        executor = Executor(self.interpreter, self.program, self.n_processes)
        executor.debug = self.debug
        executor.invoke(jobs)


if __name__ == "__main__":
    from sys import argv
    args = ArgumentParser().parse_arguments(argv[1:])
    driver = Driver(args)
    driver.run(args)
