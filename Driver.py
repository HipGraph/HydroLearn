import os
import sys
import signal
from threading import Timer
from subprocess import Popen
from importlib import import_module

from Container import Container
from Arguments import ArgumentParser, ArgumentBuilder, arg_flag
import Utility as util
import Gather


def signal_handler(sig, frame):
    print("Exiting from interrupt Ctrl+C")
    sys.exit()


class Executor:

    def __init__(
        self, 
        interpreter="python", 
        program="Execute.py", 
        n_processes=1, 
        input_timeout=60, 
        input_default="n", 
        checkpoint=True, 
        pass_as_argfile=False, 
        debug=False, 
    ):
        self.interpreter = interpreter
        self.program = program
        self.n_processes = n_processes
        self.input_timeout = input_timeout
        self.input_default = input_default
        self.checkpoint = checkpoint
        self.pass_as_argfile = pass_as_argfile
        self.debug = debug
        signal.signal(signal.SIGINT, self.on_interrupt)

    def invoke(self, jobs):
        arg_bldr, chkptr = ArgumentBuilder(), Checkpointer()
        i = -1
        while i < len(jobs) - 1:
            i += 1
            job = jobs[i]
            if self.debug:
                print(util.make_msg_block(47*"#" + " Arguments %2d " % (i) + 47*"#"))
                print(job.work)
            if self.debug:
                continue
            if self.checkpoint and job.chkpt:
                if chkptr.is_completed(job):
                    print("JOB ALREADY COMPLETED. MOVING ON...")
                    continue
            args = Container().copy(job.work)
            processes = []
            for j in range(self.n_processes):
                args.set("process_rank", j, None, ["distribution"])
                invocation = [self.interpreter, self.program]
                if self.pass_as_argfile:
                    path = os.sep.join([os.path.dirname(os.path.realpath(__file__)), "args[%d].pkl" % (j)])
                    util.to_cache(args, path)
                    invocation += ["--f", path]
                else:
                    invocation += arg_bldr.build(args)
                processes += [Popen(invocation)]
            self.interrupted, exited_correctly = False, True
            for process in processes:
                return_code = process.wait()
                exited_correctly = exited_correctly and not return_code
            if self.interrupted or not exited_correctly:
                _input = self.get_failure_input()
                if "r" in _input.lower():
                    i -= 1
                elif "q" in _input.lower():
                    break
                elif "n" in _input.lower():
                    pass
                else:
                    raise ValueError("Unknown option \"%s\". Quitting..." % (_input))
            elif self.checkpoint and job.chkpt:
                chkptr.checkpoint(job)

    def get_failure_input(self):
        print(util.make_msg_block("Job failed and will not be checkpointed", "!"))
        if self.input_timeout <= 0:
            _input = self.input_default
        elif 0:
            try:
                _input = util.input_with_timeout("Repeat, next, or quit? [r,n,q]: ", self.input_timeout)
            except util.TimeoutExpired:
                print("No input received. Taking default option \"%s\"" % (self.input_default))
                _input = self.input_default
        else:
            _input = input("Repeat, next, or quit? [r,n,q]: ")
        return _input

    def on_interrupt(self, sig, frame):
        self.interrupted = True


class Checkpointer:

    sep = "\n\n"

    def __init__(self):
        pass

    def filter_for_checkpoint(self, work):
        work = Container().copy(work)
        work.rem(["checkpoint_dir", "evaluation_dir"], must_exist=False)
        return work

    def get_checkpoint_contents(self, job):
        work = self.filter_for_checkpoint(job.work)
        return ArgumentBuilder().view(work)

    def checkpoint(self, job):
        path = job.checkpoint_path()
        chkpt_dir = job.checkpoint_dir()
        os.makedirs(chkpt_dir, exist_ok=True)
        contents = self.get_checkpoint_contents(job)
        with open(path, "a+") as f:
            f.write(contents + self.sep)

    def job_from_checkpoint(self, checkpoint):
        work = Gather.parse_vars(checkpoint)
        work = self.filter_for_checkpoint(work)
        return Job(work)

    def get_completed_jobs(self, job):
        path = job.checkpoint_path()
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            checkpoints = f.read().split(self.sep)
        checkpoints = [chkpt for chkpt in checkpoints if chkpt != ""]
        completed_jobs = [self.job_from_checkpoint(chkpt) for chkpt in checkpoints]
        return completed_jobs

    def is_completed(self, job):
        return Job(self.filter_for_checkpoint(job.work)) in self.get_completed_jobs(job)


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

    def __eq__(self, job):
        return self.work == job.work

    def __str__(self):
        return self.work.to_string(extent=[-1, -1])


class Driver:

    interpreter = "python"
    program = "Execute.py"
    n_processes = 1
    input_timeout = sys.float_info.max
    input_default = "n"
    exec_index = None
    checkpoint = True
    pass_as_argfile = False
    debug = False

    def __init__(self, args):
        driver_args = None
        if "driver" in args:
            driver_args = args.driver
            args.rem("driver")
        if not driver_args is None:
            if "interpreter" in driver_args: self.interpreter = driver_args.interpreter
            if "program" in driver_args: self.program = driver_args.program
            if "n_processes" in driver_args: self.n_processes = driver_args.n_processes
            if "input_timeout" in driver_args: self.input_timeout = driver_args.input_timeout
            if "input_default" in driver_args: self.input_default = driver_args.input_default
            if "exec_index" in driver_args: self.exec_index = driver_args.exec_index
            if "checkpoint" in driver_args: self.checkpoint = driver_args.checkpoint
            if "pass_as_argfile" in driver_args: self.pass_as_argfile = driver_args.pass_as_argfile
            if "debug" in driver_args: self.debug = driver_args.debug

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
                    exp = getattr(exp_module, "Experiment__%s" % (str(_exp_id).replace(".", "_")))()
                    jobs += self.filter_jobs(exp.jobs, self.exec_index)
            else:
                exp = getattr(exp_module, "Experiment__%s" % (str(exp_id).replace(".", "_")))()
                jobs = self.filter_jobs(exp.jobs, self.exec_index)
        elif "A" in args: # Invoking an anlysis
            if not "a" in args:
                raise ValueError("Given %sA but missing %sa" % (arg_flag, arg_flag))
            ana_module_name, ana_id = args.get(["A", "a"])
            args.rem(["A", "a"])
            ana_module = import_module("Analysis.%s" % (ana_module_name))
            if not isinstance(ana_id, list):
                ana_id = [ana_id]
            for _ana_id in ana_id:
                ana = getattr(ana_module, "Analysis__%s" % (str(_ana_id).replace(".", "_")))
                ana().run(args)
            sys.exit()
        elif "I" in args: # Invoking data integration
            integration_module = import_module("Data.Integration")
            integrator_class_name = args.get("I")
            args.rem("I")
            acquire, convert = True, True
            if "acquire" in args:
                acquire = args.get("acquire")
                args.rem("acquire")
            if "convert" in args:
                convert = args.get("convert")
                args.rem("convert")
            if not isinstance(integrator_class_name, list):
                integrator_class_name = [integrator_class_name]
            for _integrator_class_name in integrator_class_name:
                integrator_class = getattr(integration_module, _integrator_class_name.replace(".", "_"))
                integrator = integrator_class()
            if acquire:
                integrator.acquire(args)
            if convert:
                integrator.convert(args)
            sys.exit()
        elif "G" in args: # Invoking data generation
            generation_module = import_module("Data.Generation")
            generator_class_name = args.get("G")
            args.rem("G")
            if not isinstance(generator_class_name, list):
                generator_class_name = [generator_class_name]
            for _generator_class_name in generator_class_name:
                generator_class = getattr(generation_module, _generator_class_name)
                generator = generator_class()
                generator.generate(args)
            sys.exit()
        else: # Invoking a single job with supplied args
            jobs = [Job(args)]
        # Edit jobs
        for job in jobs: # Add driver-level args to all jobs
            job.work.merge(args, coincident_only=False)
        # All jobs are ready - move to invocation
        executor = Executor(
            self.interpreter, 
            self.program, 
            self.n_processes, 
            self.input_timeout, 
            self.input_default, 
            self.checkpoint, 
            self.pass_as_argfile, 
            self.debug
        )
        executor.invoke(jobs)

    def filter_jobs(self, jobs, exec_index):
        # Get index of jobs to keep
        keep_index = list(range(len(jobs)))
        if exec_index is None:
            pass
        elif exec_index[0] == "~":
            keep_index = util.list_subtract(keep_index, exec_index[1:])
        else:
            keep_index = exec_index
        # Apply filter
        return [jobs[i] for i in keep_index]


if __name__ == "__main__":
    from sys import argv
    args = ArgumentParser().parse_arguments(argv[1:])
    driver = Driver(args)
    driver.run(args)
