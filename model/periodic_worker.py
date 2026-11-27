"""Disposable process entry point for native periodic calculations."""
import json
import sys
import traceback
from pathlib import Path


def main():
    request=Path(sys.argv[1])
    payload=json.loads(request.read_text(encoding='utf-8'))
    with request.with_suffix('.jsonl').open('w',encoding='utf-8',buffering=1) as stream:
        def emit(item):
            stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
        try:
            from model.shared_eqn import Parameters
            from model.continuation_config import Axis
            from model.periodic_native import execute_task
            from model.periodic_config import periodic_options
            execute_task(payload['task'],Parameters(**payload['base']),
                         [Axis(**a) for a in payload['axes']],periodic_options(payload['options']),emit)
            emit(dict(type='complete'))
        except Exception as exc:
            emit(dict(type='failure',reason=str(exc),traceback=traceback.format_exc()))
            return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
