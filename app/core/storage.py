"""SQLite job and seed registry; GUI reads only small, paged result sets."""
from __future__ import annotations
import csv
import json
import re
import sqlite3
import uuid
import shutil
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from .paths import STATE, RUNS, ensure_dirs
from .discovery import Analysis, validate


def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def slug(s: str) -> str: return (re.sub(r'[^a-zA-Z0-9_-]+', '_', s).strip('_') or 'job')[:45]


def remove_result_tree(folder):
    """Handle OneDrive's read-only flag and brief Windows sharing locks."""
    def retry(operation, path, error):
        if not isinstance(error, PermissionError): raise error
        target = Path(path)
        # rmtree never follows directory junctions/symlinks; do not chmod targets.
        if not target.is_symlink() and not target.is_junction():
            target.chmod(target.stat().st_mode | stat.S_IWRITE)
        for attempt in range(5):
            try:
                operation(path)
                return
            except PermissionError:
                if attempt == 4: raise
                time.sleep(.05 * 2**attempt)
    shutil.rmtree(folder, onexc=retry)

def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


class ManagedConnection(sqlite3.Connection):
    """Commit/rollback AND close the SQLite handle on context exit."""
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class Repository:
    def __init__(self, db_path: Path | None = None, runs_dir: Path | None = None):
        ensure_dirs()
        self.db_path = Path(db_path) if db_path else STATE / 'jobs.sqlite'
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, analysis_id TEXT NOT NULL,
              status TEXT NOT NULL, config_json TEXT NOT NULL, output_dir TEXT NOT NULL,
              created_at TEXT NOT NULL, started_at TEXT, ended_at TEXT, exit_code INTEGER,
              progress REAL NOT NULL DEFAULT 0, phase TEXT DEFAULT '', batch_id TEXT,
              pid INTEGER, error TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status, created_at);
            CREATE TABLE IF NOT EXISTS runtime_history (
              job_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL,
              parameters_json TEXT NOT NULL, seconds REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS seeds (
              seed_id TEXT PRIMARY KEY, source_job_id TEXT NOT NULL,
              values_json TEXT NOT NULL, created_at TEXT NOT NULL,
              metrics_json TEXT NOT NULL DEFAULT '{}', bistable INTEGER, delta_s REAL
            );
            ''')
            # Non-destructive migration of the database shipped in version 1.
            cols = {row['name'] for row in db.execute('PRAGMA table_info(seeds)')}
            for column, declaration in [('metrics_json', "TEXT NOT NULL DEFAULT '{}'"),
                                        ('bistable', 'INTEGER'), ('delta_s', 'REAL'),
                                        ('seed_kind', "TEXT NOT NULL DEFAULT 'generated'")]:
                if column not in cols:
                    db.execute(f'ALTER TABLE seeds ADD COLUMN {column} {declaration}')
            db.execute('CREATE INDEX IF NOT EXISTS ix_seeds_source ON seeds(source_job_id, seed_id)')
            db.execute('CREATE INDEX IF NOT EXISTS ix_seeds_bistable ON seeds(source_job_id, bistable)')
            db.execute('CREATE INDEX IF NOT EXISTS ix_seeds_kind ON seeds(seed_kind, source_job_id)')
            if 'stages_json' not in {row['name'] for row in db.execute('PRAGMA table_info(runtime_history)')}:
                db.execute("ALTER TABLE runtime_history ADD COLUMN stages_json TEXT NOT NULL DEFAULT '{}'")
        # Repoint records when the user copied an installation to another folder.
        self.rebase_copied_results()
        # Completed jobs from older versions are reusable immediately after upgrade.
        self.backfill_completed_run_seeds()
        self.learn_completed_runtimes()

    def learn_completed_runtimes(self):
        from .runtime_estimates import measured_seconds
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM jobs WHERE status='completed'
                AND id NOT IN (SELECT job_id FROM runtime_history)""").fetchall()
            for row in rows:
                job = dict(row)
                seconds = measured_seconds(job)
                if seconds is not None:
                    parameters = json.loads(job['config_json']).get('parameters', {})
                    try:
                        stages=json.loads((Path(job['output_dir'])/'progress.json').read_text()).get('stage_seconds',{})
                    except (OSError,ValueError):stages={}
                    db.execute('INSERT OR IGNORE INTO runtime_history (job_id,analysis_id,parameters_json,seconds,stages_json) VALUES (?,?,?,?,?)',
                               (job['id'], job['analysis_id'], json.dumps(parameters), seconds,json.dumps(stages)))

    def runtime_label(self, job, paused=False):
        from .runtime_estimates import live_label
        self.learn_completed_runtimes()
        with self.connect() as db:
            history=[dict(row) for row in db.execute('SELECT * FROM runtime_history WHERE analysis_id=? ORDER BY rowid DESC LIMIT 30',
                                                    (job['analysis_id'],))]
        return live_label(job,history,paused)

    def estimate_runtime(self, job):
        from .runtime_estimates import estimate
        with self.connect() as db:
            rows = db.execute('SELECT * FROM runtime_history WHERE analysis_id=? ORDER BY rowid DESC LIMIT 100',
                              (job['analysis_id'],)).fetchall()
        return estimate(job, [dict(row) for row in rows])

    def rebase_copied_results(self):
        """Prefer copied run folders inside the current installation after an upgrade.

        The database traditionally stores absolute paths. The run ID and folder
        basename are stable, so an existing local copy can be matched safely.
        """
        with self.connect() as db:
            for row in db.execute('SELECT id,output_dir FROM jobs').fetchall():
                prior=Path(row['output_dir'])
                candidate=self.runs_dir/prior.name
                if (candidate.is_dir() and candidate.resolve()!=prior.resolve()
                        and prior.parent.name=='runs'):
                    db.execute('UPDATE jobs SET output_dir=? WHERE id=?',
                               (str(candidate),row['id']))

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=15, factory=ManagedConnection)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA busy_timeout=15000')
        return db

    def create_job(self, name: str, analysis: Analysis, parameters: dict, batch_id=None,
                   seed_id: str | None = None, extra_parameters: dict | None = None) -> dict:
        """Create immutable, validated run configuration; allow editable inherited seed fields.

        `parameters` must match the declared schema. `extra_parameters` may only
        override keys inherited from the selected seed that aren't in the schema.
        The scientific script receives ALL effective settings in config['parameters'].
        """
        name = name.strip()
        if not name or len(name) > 100: raise ValueError('Job name must contain 1–100 characters')
        selected_id = seed_id if seed_id is not None else parameters.get('seed_id', '')
        seed = None
        if selected_id:
            seed = self.get_seed(selected_id)
            if seed is None:
                raise ValueError('Selected Seed-ID is no longer present in the seed library')
        original = json.loads(seed['values_json']) if seed else {}
        # Seed values prefill unspecified manifest fields; explicit user inputs win.
        prefilled = {k:v for k,v in original.items() if k in analysis.parameters and k!='seed_id'}
        prefilled.update(parameters)
        pars = validate(analysis, prefilled)
        extras = extra_parameters or {}
        if extras and not seed:
            raise ValueError('Additional parameters require selecting an existing seed')
        invalid = set(extras) - (set(original) - set(analysis.parameters))
        if invalid:
            raise ValueError(f'Unknown inherited parameters: {", ".join(sorted(invalid))}')
        effective = dict(original)
        effective.update({key: value for key, value in pars.items() if key != 'seed_id'})
        effective.update(extras)
        if 'seed_id' in analysis.parameters:
            effective['seed_id'] = selected_id or ''
        ident = uuid.uuid4().hex
        dirname = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + slug(name) + '_' + ident[:8]
        output = self.runs_dir / dirname
        config = {'id': ident, 'name': name, 'analysis_id': analysis.id, 'parameters': effective,
                  'declared_parameters': pars, 'seed_id': selected_id or '',
                  'created_at': now(), 'batch_id': batch_id}
        if seed:
            # Freeze both ORIGINAL seed and OVERRIDDEN effective inputs for reproducibility.
            config['seed_source_job_id'] = seed['source_job_id']
            config['seed_values'] = original
            config['seed_overrides'] = {key: val for key, val in effective.items()
                                         if key in original and val != original[key]}
            config['seed_metrics'] = json.loads(seed.get('metrics_json') or '{}')
        output.mkdir(parents=True, exist_ok=False)
        dump_json(output / 'config.json', config)
        with self.connect() as db:
            db.execute('''INSERT INTO jobs
              (id,name,analysis_id,status,config_json,output_dir,created_at,batch_id)
              VALUES (?,?,?,'draft',?,?,?,?)''',
              (ident,name,analysis.id,json.dumps(config),str(output),now(),batch_id))
        return self.get_job(ident)

    def get_job(self, ident):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone()
        return dict(row) if row else None

    def delete_result(self, ident):
        """Delete a terminal run and its generated seeds; never touch active jobs.

        Hold a write lock through validation/deletion to prevent a queue race.
        Only a direct, non-linked child of this repository's runs directory is
        eligible. Other runs retain their already-saved seed snapshots.
        """
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone()
            if job is None: raise ValueError('This result no longer exists')
            if job['status'] not in ('completed', 'failed', 'stopped', 'interrupted'):
                raise ValueError('Stop the job before deleting its results')
            folder = Path(job['output_dir'])
            root = self.runs_dir.resolve()
            if folder.is_symlink() or (hasattr(folder, 'is_junction') and folder.is_junction()):
                raise ValueError('Refusing to delete a linked result directory')
            resolved = folder.resolve()
            if resolved.parent != root or resolved == root:
                raise ValueError('Result directory is outside this repository’s runs folder')
            config = resolved / 'config.json'
            if config.exists() and json.loads(config.read_text(encoding='utf-8')).get('id') != ident:
                raise ValueError('Result configuration does not match the selected run')
            if resolved.exists():
                remove_result_tree(resolved)
            db.execute('DELETE FROM seeds WHERE source_job_id=?', (ident,))
            db.execute('DELETE FROM jobs WHERE id=?', (ident,))

    def list_jobs(self, limit=1000):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?', (limit,)).fetchall()
        return [dict(x) for x in rows]

    def enqueue(self, ids):
        with self.connect() as db:
            for ident in ids:
                db.execute("UPDATE jobs SET status='queued' WHERE id=? AND status='draft'", (ident,))

    def delete_draft(self, ids):
        with self.connect() as db:
            for ident in ids:
                db.execute("DELETE FROM jobs WHERE id=? AND status='draft'", (ident,))

    def claim_next(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET status='running',started_at=?,progress=0,phase='Starting' WHERE id=?", (now(),row['id']))
        return self.get_job(row['id']) if row else None

    def update_job(self, ident, **fields):
        permitted = {'status','progress','phase','started_at','ended_at','exit_code','pid','error'}
        items = {k:v for k,v in fields.items() if k in permitted}
        if not items: return
        with self.connect() as db:
            db.execute(f'UPDATE jobs SET {", ".join(k+"=?" for k in items)} WHERE id=?', (*items.values(),ident))

    def register_seeds(self, job_id: str, csv_file: Path):
        """Stream tens/hundreds of thousands of samples into SQLite, never a UI list.

        Required CSV: seed_id + model parameter columns.
        Optional `metric__NAME` columns are outputs, NOT parameters copied to new runs.
        """
        if not csv_file.is_file(): return 0
        added = 0
        with csv_file.open(newline='', encoding='utf-8-sig') as fp, self.connect() as db:
            reader=csv.DictReader(fp)
            if not reader.fieldnames or 'seed_id' not in reader.fieldnames:
                raise ValueError('seeds.csv must have a seed_id column')
            for record in reader:
                ident = (record.pop('seed_id', '') or '').strip()
                if not ident: continue
                values, metrics = {}, {}
                for key, raw in record.items():
                    if not key or raw is None or raw == '': continue
                    try:
                        val = float(raw)
                        if val.is_integer(): val = int(val)
                    except (ValueError, TypeError): val = raw
                    if key.startswith('metric__'): metrics[key[8:]] = val
                    else: values[key] = val
                bistable = metrics.get('bistability')
                delta_s = metrics.get('delta_S')
                cur = db.execute('''INSERT INTO seeds
                     (seed_id, source_job_id, values_json, created_at, metrics_json, bistable, delta_s, seed_kind)
                     VALUES (?,?,?,?,?,?,?,'generated')
                     ON CONFLICT(seed_id) DO UPDATE SET
                       values_json=excluded.values_json, metrics_json=excluded.metrics_json,
                       bistable=excluded.bistable, delta_s=excluded.delta_s
                     WHERE seeds.source_job_id=excluded.source_job_id''',
                     (ident,job_id,json.dumps(values),now(),json.dumps(metrics),
                      int(bistable) if bistable in (0,1,True,False) else None,
                      float(delta_s) if isinstance(delta_s,(int,float)) else None))
                if cur.rowcount == 0:
                    raise ValueError(f'Seed-ID {ident!r} already belongs to another run')
                added += 1
        return added


    @staticmethod
    def _run_seed_values(config):
        """Snapshot the effective parameters, including inherited model parameters.

        `seed_id` is a provenance pointer, never a model parameter. Empty strings
        are omitted. New run controls override same-named source seed values.
        """
        vals = dict(config.get('seed_values') or {})
        for key, value in (config.get('parameters') or {}).items():
            if key != 'seed_id' and value != '':
                vals[key] = value
        return vals

    def register_run_seed(self, job_id: str) -> bool:
        """Every *completed* analysis becomes a reusable seed, regardless of type.

        Deterministic ID ensures repeated app launches never duplicate it.
        """
        with self.connect() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=? AND status='completed'",(job_id,)).fetchone()
            if not job: return False
            config = json.loads(job['config_json'])
            params = self._run_seed_values(config)
            if not params: return False
            metrics = {'source_analysis': job['analysis_id'], 'origin': 'completed_run'}
            cur = db.execute("""INSERT OR IGNORE INTO seeds
                 (seed_id,source_job_id,values_json,created_at,metrics_json,bistable,delta_s,seed_kind)
                 VALUES (?,?,?,?,?,NULL,NULL,'run')""",
                 ('RUN-'+job_id,job_id,json.dumps(params),job['ended_at'] or job['created_at'],json.dumps(metrics)))
            return cur.rowcount == 1

    def backfill_completed_run_seeds(self) -> int:
        """Index earlier completed jobs, preserving existing seeds and settings."""
        with self.connect() as db:
            rows = db.execute("""SELECT j.id,j.analysis_id,j.config_json,j.created_at,j.ended_at
                FROM jobs j LEFT JOIN seeds s ON s.seed_id=('RUN-' || j.id)
                WHERE j.status='completed' AND s.seed_id IS NULL""").fetchall()
            count = 0
            for job in rows:
                try:
                    params = self._run_seed_values(json.loads(job['config_json']))
                except (ValueError,TypeError,AttributeError):
                    continue
                if not params: continue
                meta = {'source_analysis':job['analysis_id'], 'origin':'completed_run'}
                cur = db.execute("""INSERT OR IGNORE INTO seeds
                     (seed_id,source_job_id,values_json,created_at,metrics_json,bistable,delta_s,seed_kind)
                     VALUES (?,?,?,?,?,NULL,NULL,'run')""",
                     ('RUN-'+job['id'],job['id'],json.dumps(params),
                      job['ended_at'] or job['created_at'],json.dumps(meta)))
                count += cur.rowcount
        return count

    def get_seed(self, seed_id):
        with self.connect() as db:
            row = db.execute('''SELECT s.*, COALESCE(j.name,'(archived source)') AS source_name
                FROM seeds s LEFT JOIN jobs j ON j.id=s.source_job_id WHERE seed_id=?''',
                (seed_id,)).fetchone()
        return dict(row) if row else None

    def list_seed_sources(self, limit=400):
        """Return source RUNS, not thousands of individual Seed-IDs."""
        with self.connect() as db:
            rows=db.execute('''SELECT s.source_job_id, COALESCE(j.name,'(archived source)') AS source_name,
                  COALESCE(j.analysis_id,'') AS source_analysis_id,
                  COUNT(*) AS seed_count, MAX(s.created_at) AS created_at
                  FROM seeds s LEFT JOIN jobs j ON j.id=s.source_job_id
                  GROUP BY s.source_job_id ORDER BY created_at DESC LIMIT ?''',(limit,)).fetchall()
        return [dict(x) for x in rows]

    def seed_count(self, source_job_id=None):
        with self.connect() as db:
            if source_job_id:
                row=db.execute('SELECT COUNT(*) AS n FROM seeds WHERE source_job_id=?',(source_job_id,)).fetchone()
            else: row=db.execute('SELECT COUNT(*) AS n FROM seeds').fetchone()
        return row['n']

    def find_seeds(self, source_job_id=None, query='', bistability='all', limit=80, offset=0, seed_kind='all'):
        """SQL-paginated seed search, indexed by source; safe with very large sweeps."""
        limit=max(1,min(int(limit),200));offset=max(0,int(offset))
        where=[];args=[]
        if source_job_id:
            where.append('s.source_job_id=?');args.append(source_job_id)
        if query.strip():
            where.append('(s.seed_id LIKE ?)');args.append('%'+query.strip()+'%')
        if bistability in ('yes','no'):
            where.append('s.bistable=?');args.append(1 if bistability=='yes' else 0)
        if seed_kind in ('run','generated'):
            where.append('s.seed_kind=?');args.append(seed_kind)
        filter_sql='WHERE '+' AND '.join(where) if where else ''
        with self.connect() as db:
            total=db.execute('SELECT COUNT(*) FROM seeds s '+filter_sql,args).fetchone()[0]
            rows=db.execute('''SELECT s.*, COALESCE(j.name,'(archived source)') AS source_name
                FROM seeds s LEFT JOIN jobs j ON j.id=s.source_job_id '''+filter_sql+
                ''' ORDER BY s.created_at DESC, s.seed_id ASC LIMIT ? OFFSET ?''',
                [*args,limit,offset]).fetchall()
        return total,[dict(x) for x in rows]

    def list_seeds(self,limit=2000):
        # Compatibility helper for automated tests or small scripts; never call in GUI.
        with self.connect() as db:
            rows=db.execute('SELECT * FROM seeds ORDER BY created_at DESC,seed_id LIMIT ?', (limit,)).fetchall()
        return [dict(x) for x in rows]
