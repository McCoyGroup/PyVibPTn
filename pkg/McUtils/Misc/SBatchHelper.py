"""
Provides minimal utilities for making slurm stuff nicer
"""

import inspect

__all__ = ["SBatchJob"]

class SBatchJob:
    """
    Provides a simple interface to formatting SLURM
    files so that they can be submitted to `sbatch`.
    The hope is that this can be subclassed codify
    options for different HPC paritions and whatnot.
    """
    slurm_keys = [
        "account", "acctg-freq", "array", "batch", "bb", "bbf", "begin", "chdir",
        "cluster-constraint", "clusters", "comment", "constraint", "contiguous",
        "core-spec", "cores-per-socket", "cpu-freq", "cpus-per-gpu", "cpus-per-task",
        "deadline", "delay-boot", "dependency", "distribution", "error", "exclude",
        "exclusive", "export", "export-file", "extra-node-info",
        "get-user-env", "gid", "gpu-bind", "gpu-freq", "gpus", "gpus-per-node",
        "gpus-per-socket", "gpus-per-task", "gres", "gres-flags", "help", "hint",
        "ignore-pbs", "input", "job-name", "kill-on-invalid-dep", "licenses",
        "mail-type", "mail-user", "mcs-label", "mem", "mem-bind", "mem-per-cpu",
        "mem-per-gpu", "mincpus", "network", "nice", "nodefile", "nodelist", "nodes",
        "no-kill", "no-requeue", "ntasks", "ntasks-per-core", "ntasks-per-gpu",
        "ntasks-per-node", "ntasks-per-socket", "open-mode", "output", "overcommit",
        "oversubscribe", "parsable", "partition", "power", "priority", "profile",
        "propagate", "qos", "quiet", "reboot", "requeue", "reservation", "signal",
        "sockets-per-node", "spread-job", "switches", "test-only", "thread-spec",
        "threads-per-core", "time", "time-min", "tmp", "uid", "usage", "use-min-nodes",
        "verbose", "version", "wait", "wait-all-nodes", "wckey", "wrap"
    ]
    default_opts = {
        'chdir': "."
    }
    def __init__(self,
                 description=None,
                 job_name=None, account=None, partition=None,
                 mem=None,  nodes=None, ntasks_per_node=None,
                 chdir=None, output=None,
                 steps=(),
                 **opts
                 ):
        self.description=description
        self.steps=steps

        base_opts = dict(
            job_name=job_name, account=account, partition=partition,
            mem=mem, nodes=nodes, ntasks_per_node=ntasks_per_node,
            output=output, chdir=chdir
        )
        base_opts = self.clean_opts(base_opts)
        base_opts = dict(self.default_opts, **base_opts)

        opts = dict(opts, **base_opts)
        self.opts = self.clean_opts(opts)

    def clean_opts(self, opts):
        """
        Makes sure opt names are clean.
        Does no validation of the values sent in.

        :param opts:
        :type opts:
        :return:
        :rtype:
        """
        clean_opts = {}
        for opt_name, opt_val in opts.items():
            opt_name = opt_name.replace("_", "-")
            if opt_val is not None:
                if opt_name not in self.slurm_keys:
                    raise ValueError("SBATCH option {} invalid; accepted ones are {}".format(opt_name, self.slurm_keys))
                clean_opts[opt_name] = opt_val
        return clean_opts

    sbatch_opt_template="#SBATCH --{name}={value}"
    def format_opt_block(self):
        """
        Formats block of options
        :return:
        :rtype:
        """
        return "\n".join(
            self.sbatch_opt_template.format(name=k, value=v) for k,v in self.opts.items() if v is not None
        )

    sbatch_template="#!/bin/bash\n{opts}\n\n{enter}\n{call}\n{exit}"
    sbatch_enter_command="\n".join([
        'echo "Starting Job $SLURM_JOB_NAME"',
        'START=$(date +%s.%N)',
        'echo "  START: $(date)"',
        'echo "    PWD: $PWD"',
        'echo "  NODES: $SLURM_JOB_NUM_NODES"',
        'echo "  PART.: $SLURM_JOB_PARTITION"',
        'echo "{sep}"'.format(sep="="*50)
        ])
    sbatch_exit_command = "\n".join([
        'echo "{sep}"'.format(sep="=" * 50),
        'END=$(date +%s.%N)',
        'DIFF=$(echo "$END - $START" | bc)',
        'echo "   END: $(date)"',
        'echo "  TIME: $DIFF"'
    ])
    def format(self):
        """
        Formats an SBATCH file from the held options
        :param call_steps:
        :type call_steps:
        :return:
        :rtype:
        """
        opts = self.format_opt_block()
        enter = self.sbatch_enter_command
        if self.description is not None:
            enter += (
                    '\necho "'
                    + inspect.cleandoc(self.description).replace("\n", '"\necho "')
                    + '"\necho\n\n'
            )
        exit = self.sbatch_exit_command

        call = "\n".join(self.steps)

        return self.sbatch_template.format(
            opts=opts,
            call=call,
            enter=enter,
            exit=exit
        )