from app.core.worker import work_loop
import argparse
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--once',action='store_true',help='Process one queued job and exit (for testing)')
    args=p.parse_args()
    raise SystemExit(work_loop(args.once))
