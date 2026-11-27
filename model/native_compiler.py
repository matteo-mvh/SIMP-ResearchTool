"""Project-owned pyoomph compiler adapter; do not patch installed packages."""
import os
import subprocess
from pyoomph.generic.ccompiler import TCCBoxCompiler


class QuietTCCCompiler(TCCBoxCompiler):
    compiler_id = 'simp_quiet_tcc'

    @staticmethod
    def call_cmd(cmd, shell=False, env=None, quiet=False):
        command = list(cmd)
        if os.name == 'nt' and command[1:3] == ['-m', 'tccbox']:
            # Bypass the Python wrapper, which starts another visible subprocess
            # and discards the compiler's exit code. Windows requires no extra
            # wrapper include/library arguments (unlike Linux/macOS).
            from tccbox import tcc_bin_path
            command = [str(tcc_bin_path()), *command[3:]]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                shell=shell, env=env, timeout=120,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        output = result.stdout.decode('utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError(f'Native compilation failed ({result.returncode}): {output}')
        return output
