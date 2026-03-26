import asyncio
import atexit
import getpass
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DEFAULT_TIMEOUT = 60.0
DEFAULT_JULIA_ARGS = ("--startup-file=no", "--threads=auto")
PKG_PATTERN = re.compile(r"\bPkg\.")
TEMP_SESSION_KEY = "__temp__"
SENTINEL_BASE = Path(tempfile.gettempdir()) / ".julia-mcp-jobs" / getpass.getuser()
SERVER_UUID = uuid.uuid4().hex[:16]


@dataclass
class BackgroundJob:
    job_id: str
    env_path: str | None
    started_at: float
    lines: list[str]
    status: str  # "running" | "completed" | "error"
    result: str | None
    error: str | None
    reader_task: asyncio.Task | None
    delivered: bool
    sentinel_path: Path


mcp = FastMCP("julia")


@dataclass
class _TimeoutResult:
    """Internal: returned when _execute_raw times out instead of completing."""
    reader_task: asyncio.Task
    lines: list[str]
    timeout: float


class JuliaSession:
    def __init__(
        self,
        env_dir: str,
        sentinel: str,
        *,
        is_temp: bool = False,
        is_test: bool = False,
        julia_args: tuple[str, ...] = DEFAULT_JULIA_ARGS,
        log_file: TextIOWrapper | None = None,
    ):
        self.env_dir = env_dir
        self.sentinel = sentinel
        self.is_temp = is_temp
        self.is_test = is_test
        self.julia_args = julia_args
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self._log_file = log_file
        self._background_job: BackgroundJob | None = None

    @property
    def project_path(self) -> str:
        if self.is_test:
            return str(Path(self.env_dir).parent)
        return self.env_dir

    @property
    def init_code(self) -> str | None:
        if self.is_test:
            return "using TestEnv; TestEnv.activate()"
        return None

    async def start(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            raise RuntimeError(
                "Julia not found in PATH. Install from https://julialang.org/downloads/"
            )

        cmd = [
            julia,
            "-i",
            *self.julia_args,
            f"--project={self.project_path}",
        ]

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.env_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            limit=64 * 1024 * 1024,  # 64 MB readline buffer
        )

        # Wait for readiness
        await self._execute_raw(
            "",
            timeout=120.0,  # generous startup timeout
        )

        # Auto-load Revise so code changes are picked up without restarting
        await self._execute_raw(
            "try; using Revise; catch; end",
            timeout=120.0,
        )

        if self.init_code:
            await self._execute_raw(self.init_code, timeout=None)

    def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def execute(self, code: str, timeout: float | None) -> str:
        # Busy check: before the lock, return immediately
        if self._background_job is not None:
            job = self._background_job
            return (
                f"Session busy: background job {job.job_id} is running "
                f"(started {int(time.time() - job.started_at)}s ago). "
                f"Use julia_job_status(\"{job.job_id}\") to check progress, "
                f"or julia_job_cancel(\"{job.job_id}\") to abort. "
                f"For independent work, use a background Bash julia command."
            )

        async with self.lock:
            if not self.is_alive():
                raise RuntimeError("Julia session has died unexpectedly")

            hex_encoded = code.encode().hex()
            wrapped = (
                f'try; Revise.revise(); catch; end;'
                f'include_string(Main, String(hex2bytes("{hex_encoded}")));'
                f'nothing'
            )
            if self._log_file:
                ts = time.strftime("%H:%M:%S")
                self._log_file.write(f"[{ts}] julia> {code}\n")
                self._log_file.flush()

            # timeout=0: background immediately (inside the lock -- we write to stdin)
            if timeout is not None and timeout == 0:
                assert self.process.stdin is not None
                sentinel_cmd = (
                    f'flush(stderr); write(stdout, "\\n"); '
                    f'println(stdout, "{self.sentinel}"); flush(stdout)'
                )
                payload = wrapped + "\n" + sentinel_cmd + "\n"
                self.process.stdin.write(payload.encode())
                await self.process.stdin.drain()

                lines: list[str] = []
                return self._start_background_job(
                    lines, lines, "immediate background (timeout=0)"
                )

            # Normal path: foreground with possible auto-background
            result = await self._execute_raw(wrapped, timeout)

            if isinstance(result, _TimeoutResult):
                try:
                    return self._start_background_job(
                        result.reader_task, result.lines,
                        f"auto-backgrounded after {result.timeout}s",
                    )
                except Exception:
                    result.reader_task.cancel()
                    try:
                        await result.reader_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise

            if self._log_file and result:
                self._log_file.write(f"{result}\n\n")
                self._log_file.flush()
            return result

    async def _read_until_sentinel(self, lines: list[str]) -> str:
        """Read stdout lines until the sentinel marker appears."""
        while True:
            raw = await self.process.stdout.readline()
            if not raw:
                collected = "\n".join(lines)
                raise RuntimeError(
                    f"Julia process died during execution.\n"
                    f"Output before death:\n{collected}"
                )
            line = raw.decode().rstrip("\n").rstrip("\r")
            if line == self.sentinel:
                break
            lines.append(line)
        if lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def _write_sentinel(self, job: BackgroundJob) -> None:
        """Atomically write job result to sentinel file."""
        if job.status == "completed":
            content = f"SUCCESS\n{job.result or ''}"
        else:
            content = f"ERROR\n{job.error or ''}"
        try:
            tmp_path = job.sentinel_path.with_suffix(".tmp")
            tmp_path.write_text(content)
            tmp_path.rename(job.sentinel_path)
        except (FileNotFoundError, OSError):
            pass  # Sentinel dir already cleaned up (e.g., during shutdown)

    def _start_background_job(
        self,
        reader_task_or_lines: asyncio.Task | list[str],
        lines: list[str],
        reason: str,
    ) -> str:
        """Create a BackgroundJob wrapping an existing or new reader task."""
        job_id = uuid.uuid4().hex[:8]
        manager._sentinel_dir.mkdir(parents=True, exist_ok=True)
        sentinel_path = manager._sentinel_dir / f"{job_id}.sentinel"
        output_path = manager._sentinel_dir / f"{job_id}.log"
        # Create the output file immediately so tail -f can attach
        output_path.touch()
        job = BackgroundJob(
            job_id=job_id,
            env_path=None if self.is_temp else self.env_dir,
            started_at=time.time(),
            lines=lines,
            status="running",
            result=None,
            error=None,
            reader_task=None,
            delivered=False,
            sentinel_path=sentinel_path,
        )

        if isinstance(reader_task_or_lines, asyncio.Task):
            existing_reader = reader_task_or_lines
        else:
            existing_reader = asyncio.create_task(
                self._read_until_sentinel(reader_task_or_lines)
            )

        async def _flush_output():
            """Periodically flush accumulated lines to the output file."""
            flushed = 0
            while job.status == "running":
                current = len(job.lines)
                if current > flushed:
                    try:
                        with open(output_path, "a") as f:
                            for line in job.lines[flushed:current]:
                                f.write(line + "\n")
                    except OSError:
                        pass
                    flushed = current
                await asyncio.sleep(1)
            # Final flush
            current = len(job.lines)
            if current > flushed:
                try:
                    with open(output_path, "a") as f:
                        for line in job.lines[flushed:current]:
                            f.write(line + "\n")
                except OSError:
                    pass

        async def _run_bg():
            flusher = asyncio.create_task(_flush_output())
            try:
                output = await existing_reader
                job.result = output
                job.status = "completed"
            except asyncio.CancelledError:
                job.error = "Cancelled"
                job.status = "error"
                raise
            except RuntimeError as e:
                job.error = str(e)
                job.status = "error"
            finally:
                await flusher
                self._write_sentinel(job)
                self._background_job = None
                if self._log_file:
                    ts = time.strftime("%H:%M:%S")
                    status_str = job.status.upper()
                    out = job.result or job.error or ""
                    self._log_file.write(
                        f"[{ts}] [BG {job.job_id} {status_str}] {out}\n\n"
                    )
                    self._log_file.flush()

        job.reader_task = asyncio.create_task(_run_bg())
        self._background_job = job
        manager._completed_jobs[job_id] = job

        if self._log_file:
            ts = time.strftime("%H:%M:%S")
            self._log_file.write(f"[{ts}] [BG {job.job_id} STARTED] {reason}\n")
            self._log_file.flush()

        return f"[BACKGROUNDED] job_id={job_id} sentinel={sentinel_path}"

    async def _execute_raw(
        self, code: str, timeout: float | None
    ) -> str | _TimeoutResult:
        assert self.process is not None
        assert self.process.stdin is not None

        sentinel_cmd = (
            f'flush(stderr); write(stdout, "\\n"); '
            f'println(stdout, "{self.sentinel}"); flush(stdout)'
        )
        payload = code + "\n" + sentinel_cmd + "\n"
        self.process.stdin.write(payload.encode())
        await self.process.stdin.drain()

        lines: list[str] = []

        if timeout is not None and timeout > 0:
            reader_task = asyncio.create_task(
                self._read_until_sentinel(lines)
            )
            timer_task = asyncio.create_task(asyncio.sleep(timeout))

            done, pending = await asyncio.wait(
                {reader_task, timer_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if reader_task in done:
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass
                return reader_task.result()  # may raise RuntimeError if process died
            else:
                # Timer fired first -- reader is still alive
                return _TimeoutResult(
                    reader_task=reader_task,
                    lines=lines,
                    timeout=timeout,
                )
        else:
            return await self._read_until_sentinel(lines)

    async def kill(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        if self.is_temp and os.path.isdir(self.env_dir):
            shutil.rmtree(self.env_dir, ignore_errors=True)


class SessionManager:
    def __init__(self, julia_args: tuple[str, ...] = DEFAULT_JULIA_ARGS):
        self.julia_args = julia_args
        self._sessions: dict[str, JuliaSession] = {}
        self._create_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._log_dir = tempfile.mkdtemp(prefix="julia-mcp-logs-")
        self._log_files: dict[str, TextIOWrapper] = {}
        self._sentinel_dir = SENTINEL_BASE / SERVER_UUID
        self._sentinel_dir.mkdir(parents=True, exist_ok=True)
        self._completed_jobs: dict[str, BackgroundJob] = {}
        # Clean up stale sentinel dirs from crashed previous runs
        try:
            for stale in SENTINEL_BASE.iterdir():
                if stale != self._sentinel_dir and stale.is_dir():
                    try:
                        shutil.rmtree(stale)
                    except OSError:
                        pass
        except FileNotFoundError:
            pass
        atexit.register(self._cleanup_logs)

    def _get_log_file(self, key: str) -> TextIOWrapper:
        if key not in self._log_files:
            safe_name = key.replace("/", "_").replace("\\", "_").strip("_") or "temp"
            path = os.path.join(self._log_dir, f"{safe_name}.log")
            self._log_files[key] = open(path, "a")
        return self._log_files[key]

    def _cleanup_logs(self) -> None:
        for f in self._log_files.values():
            try:
                f.close()
            except Exception:
                pass
        shutil.rmtree(self._log_dir, ignore_errors=True)

    def _key(self, env_path: str | None) -> str:
        if env_path is None:
            return TEMP_SESSION_KEY
        return str(Path(env_path).resolve())

    async def get_or_create(self, env_path: str | None) -> JuliaSession:
        key = self._key(env_path)

        # Fast path
        if key in self._sessions and self._sessions[key].is_alive():
            return self._sessions[key]

        # Get per-key creation lock
        async with self._global_lock:
            if key not in self._create_locks:
                self._create_locks[key] = asyncio.Lock()
            create_lock = self._create_locks[key]

        async with create_lock:
            # Double-check
            if key in self._sessions and self._sessions[key].is_alive():
                return self._sessions[key]

            # Clean up dead session
            if key in self._sessions:
                await self._sessions[key].kill()
                del self._sessions[key]

            # Create new session
            sentinel = f"__JULIA_MCP_{uuid.uuid4().hex}__"
            is_temp = env_path is None
            if is_temp:
                env_dir = tempfile.mkdtemp(prefix="julia-mcp-")
                is_test = False
            else:
                resolved = Path(env_path).resolve()
                env_dir = str(resolved)
                is_test = resolved.name == "test"

            session = JuliaSession(
                env_dir, sentinel, is_temp=is_temp, is_test=is_test,
                julia_args=self.julia_args,
                log_file=self._get_log_file(key),
            )
            await session.start()
            self._sessions[key] = session
            return session

    async def restart(self, env_path: str | None) -> None:
        key = self._key(env_path)
        if key in self._sessions:
            await self._sessions[key].kill()
            del self._sessions[key]

    def list_sessions(self) -> list[dict]:
        result = []
        for key, session in self._sessions.items():
            info = {
                "env_path": session.env_dir,
                "alive": session.is_alive(),
                "temp": session.is_temp,
            }
            if key in self._log_files:
                info["log_file"] = self._log_files[key].name
            result.append(info)
        return result

    async def shutdown(self) -> None:
        for session in self._sessions.values():
            await session.kill()
        self._sessions.clear()
        shutil.rmtree(self._sentinel_dir, ignore_errors=True)
        self._completed_jobs.clear()
        self._cleanup_logs()


manager = SessionManager()


@mcp.tool()
async def julia_eval(
    code: str,
    env_path: str | None = None,
    timeout: float | None = None,
) -> str:
    """Execute Julia code in a persistent REPL session.

    Each env_path gets its own session, started lazily. State persists between calls.
    Long-running jobs auto-background on timeout instead of being killed.

    Args:
        code: Julia code to evaluate. Use display(...)/println(...) to see output.
        env_path: Julia project directory path. Omit for a temporary environment.
        timeout: Seconds before auto-backgrounding (default: 60).
            Set to 0 to background immediately.
            Auto-disabled for Pkg operations.
    """
    if timeout is None:
        effective_timeout: float | None = (
            None if PKG_PATTERN.search(code) else DEFAULT_TIMEOUT
        )
    elif timeout == 0:
        effective_timeout = 0  # immediate background
    else:
        effective_timeout = timeout if timeout > 0 else None

    try:
        session = await manager.get_or_create(env_path)
        output = await session.execute(code, timeout=effective_timeout)
        return output if output else "(no output)"
    except RuntimeError as e:
        # Clean up dead session so next call starts fresh
        key = manager._key(env_path)
        if key in manager._sessions and not manager._sessions[key].is_alive():
            del manager._sessions[key]
        return f"Error: {e}"


@mcp.tool()
async def julia_job_status(job_id: str) -> str:
    """Check status of a background Julia job.

    Args:
        job_id: The job ID returned by julia_eval when a job was backgrounded.

    Returns:
        Job status (running/completed/error) and output.
    """
    job = manager._completed_jobs.get(job_id)
    if job is None:
        return f"Error: job '{job_id}' not found."

    elapsed = int(time.time() - job.started_at)

    if job.status == "running":
        partial = "\n".join(job.lines)
        msg = f"Status: running ({elapsed}s elapsed)"
        if partial:
            msg += f"\n\nPartial output:\n{partial}"
        return msg
    elif job.status == "completed":
        job.delivered = True
        msg = f"Status: completed ({elapsed}s elapsed)"
        if job.result:
            msg += f"\n\nOutput:\n{job.result}"
        return msg
    else:  # error
        job.delivered = True
        msg = f"Status: error ({elapsed}s elapsed)"
        if job.error:
            msg += f"\n\nError:\n{job.error}"
        return msg


@mcp.tool()
async def julia_job_cancel(job_id: str) -> str:
    """Cancel a running background Julia job.

    Sends interrupt signal to Julia. Session stays alive for future use.

    Args:
        job_id: The job ID to cancel.
    """
    job = manager._completed_jobs.get(job_id)
    if job is None:
        return f"Error: job '{job_id}' not found."
    if job.status != "running":
        return f"Job '{job_id}' is already {job.status}."

    # Find the session that owns this job
    key = manager._key(job.env_path)
    session = manager._sessions.get(key)
    if session is None or not session.is_alive():
        return f"Error: session for job '{job_id}' is no longer available."

    async with session.lock:
        try:
            session.process.send_signal(signal.SIGINT)
            await asyncio.sleep(0.1)
            session.process.send_signal(signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

    # Wait for the reader task to finish (up to 10s)
    if job.reader_task is not None:
        try:
            await asyncio.wait_for(job.reader_task, timeout=10.0)
        except asyncio.TimeoutError:
            # Reader didn't finish -- kill the process
            try:
                session.process.kill()
                await session.process.wait()
            except (ProcessLookupError, OSError):
                pass
            job.status = "error"
            job.error = "Cancelled (process killed after interrupt failed)"
            session._write_sentinel(job)
            session._background_job = None

    partial = "\n".join(job.lines)
    msg = f"Job '{job_id}' cancelled."
    if partial:
        msg += f"\n\nPartial output:\n{partial}"
    return msg


@mcp.tool()
async def julia_restart(env_path: str | None = None) -> str:
    """Restart a Julia session, clearing all state.

    IMPORTANT: Restarting is slow and loses all session state. Very rarely needed.
    Revise.jl is loaded automatically in every session, so code changes to loaded packages are picked up without restarting.
    Only restart as a last resort when the session is truly broken, or code changes that Revise cannot fix.
    Do NOT restart just because source files were edited between script or test runs — Revise picks up those changes automatically.

    Args:
        env_path: Environment to restart. If omitted, restarts the temporary session.
    """
    await manager.restart(env_path)
    return "Session restarted. A fresh session will start on next julia_eval call."


@mcp.tool()
async def julia_list_sessions() -> str:
    """List all active Julia sessions and their environments."""
    sessions = manager.list_sessions()
    if not sessions:
        return "No active Julia sessions."
    lines = []
    for s in sessions:
        status = "alive" if s["alive"] else "dead"
        label = f"{s['env_path']} (temp)" if s["temp"] else s["env_path"]
        log = f" log={s['log_file']}" if "log_file" in s else ""
        lines.append(f"  {label}: {status}{log}")
    return "Active Julia sessions:\n" + "\n".join(lines)


def main():
    global manager
    julia_args = tuple(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_JULIA_ARGS
    manager = SessionManager(julia_args=julia_args)
    print(f"Julia MCP log directory: {manager._log_dir}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
