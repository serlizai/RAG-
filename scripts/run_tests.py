"""统一测试入口：unit（默认离线）、integration（测试数据库）、smoke（在线 API）。"""

import argparse
from contextlib import contextmanager, ExitStack
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TEST_KEYS = ("NEO4J_KG_TEST_URI", "NEO4J_KG_TEST_PASSWORD", "MILVUS_KG_TEST_URI")
PROJECT_PREFIXES = ("OPENAI_", "LLM_", "MCP_", "MINERU_", "MILVUS_", "NEO4J_", "MINIO_", "MONGO_", "BGE_", "LOG_")
PROJECT_KEYS = {"PROJECT_ROOT", "VL_MODEL", "CHUNKS_COLLECTION", "ENTITY_NAME_COLLECTION", "ITEM_NAME_COLLECTION"}
UNIT_ENV = {
    "OPENAI_API_KEY": "unit-test-key",
    "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
    "LLM_DEFAULT_MODEL": "unit-test-model",
    "LLM_DEFAULT_TEMPERATURE": "0",
    "VL_MODEL": "unit-test-vision",
    "MCP_DASHSCOPE_BASE_URL_STREAMABLE": "http://127.0.0.1:1/mcp",
    "MINERU_BASE_URL": "http://127.0.0.1:1",
    "MINERU_API_TOKEN": "unit-test-token",
    "MILVUS_URL": "http://127.0.0.1:1",
    "NEO4J_URI": "bolt://127.0.0.1:1",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "unit-test-password",
    "NEO4J_DATABASE": "neo4j",
    "MONGO_URL": "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50&connectTimeoutMS=50",
    "MONGO_DB_NAME": "rag_unit",
    "MINIO_ENDPOINT": "127.0.0.1:1",
    "MINIO_ACCESS_KEY": "unit-test-access",
    "MINIO_SECRET_KEY": "unit-test-secret",
    "MINIO_BUCKET_NAME": "rag-unit",
    "MINIO_IMG_DIR": "/upload-images",
    "MINIO_SECURE": "False",
    "CHUNKS_COLLECTION": "rag_unit_chunks",
    "ITEM_NAME_COLLECTION": "rag_unit_products",
    "ENTITY_NAME_COLLECTION": "rag_unit_entities",
    "BGE_DEVICE": "cpu",
    "BGE_FP16": "0",
    "BGE_RERANKER_DEVICE": "cpu",
    "BGE_RERANKER_FP16": "0",
    "BGE_M3_PATH": str(ROOT / "__unit_models_not_loaded__"),
    "BGE_RERANKER_LARGE": str(ROOT / "__unit_models_not_loaded__"),
}


class ConfigurationError(ValueError):
    pass


class NetworkAccessBlocked(RuntimeError):
    pass


def _project_key(name):
    return name in PROJECT_KEYS or name.startswith(PROJECT_PREFIXES)


def _read_env(path):
    if not path.is_file():
        raise ConfigurationError(f"配置文件不存在：{path}")
    from dotenv import dotenv_values
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def build_environment(scope, *, environ=None, env_file=None):
    source = dict(os.environ if environ is None else environ)
    env = {key: value for key, value in source.items() if not _project_key(key)}
    if scope == "unit":
        if env_file is not None:
            raise ConfigurationError("unit 不读取配置文件；请使用独立的虚拟测试配置")
        env.update(UNIT_ENV)
    else:
        values = {key: value for key, value in source.items() if _project_key(key)}
        if env_file is not None:
            values.update(_read_env(Path(env_file).expanduser().resolve()))
        if scope == "integration":
            env.update(UNIT_ENV)
            env.update({key: values[key] for key in TEST_KEYS if values.get(key)})
        else:
            env.update({key: value for key, value in values.items() if _project_key(key)})
    # 所有配置在创建子进程前明确装载，应用模块不得再隐式读取业务 .env。
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PROJECT_ROOT"] = str(ROOT)
    env["LOG_CONSOLE_ENABLE"] = "False"
    env["LOG_FILE_ENABLE"] = "False"
    if scope != "smoke":
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def validate_environment(scope, env, check="model", listing=False):
    if listing or scope == "unit":
        return
    required = TEST_KEYS if scope == "integration" else ()
    if scope == "smoke":
        if check in ("model", "all"):
            required += ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_DEFAULT_TEMPERATURE")
        if check in ("mineru", "all"):
            required += ("MINERU_BASE_URL", "MINERU_API_TOKEN")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ConfigurationError("缺少显式测试配置：" + ", ".join(missing))


@contextmanager
def offline_network_guard():
    """阻止离线用例联网；即使业务代码吞掉异常，记录也会使测试入口失败。"""
    attempts = []
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def deny(kind):
        attempts.append(kind)
        raise NetworkAccessBlocked(f"离线测试禁止网络访问：{kind}")

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            return deny("socket.connect")
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            return deny("socket.connect_ex")
        return connect_ex(sock, address)

    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", guarded_connect))
        stack.enter_context(patch.object(socket.socket, "connect_ex", guarded_connect_ex))
        stack.enter_context(patch.object(socket, "create_connection", side_effect=lambda *a, **k: deny("socket.create_connection")))
        # gRPC 使用原生 socket，需同时阻止 Milvus 的公开连接入口。
        from pymilvus import connections
        stack.enter_context(patch.object(connections, "connect", side_effect=lambda *a, **k: deny("Milvus.connect")))
        yield attempts


def select_suite(suite, scope):
    selected = unittest.TestSuite()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            selected.addTests(select_suite(test, scope))
        elif getattr(test, "test_scope", "unit") == scope:
            selected.addTest(test)
    return selected


def run_unittest(scope, *, listing=False, verbose=False, failfast=False):
    loader = unittest.TestLoader()
    discovered = loader.discover(str(ROOT / "app/test"), pattern="test_*.py", top_level_dir=str(ROOT))
    if loader.errors:
        for error in loader.errors:
            print(error, file=sys.stderr)
        return 1
    suite = select_suite(discovered, scope)
    count = suite.countTestCases()
    if not count:
        print(f"未发现 {scope} 用例，拒绝把空测试集记为通过。", file=sys.stderr)
        return 2
    print(f"[{scope}] {count} tests", flush=True)
    if listing:
        def ids(tests):
            for test in tests:
                if isinstance(test, unittest.TestSuite):
                    yield from ids(test)
                else:
                    yield test.id()
        print("\n".join(ids(suite)))
        return 0
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1, failfast=failfast, buffer=True).run(suite)
    if scope == "integration" and result.skipped:
        print("integration 存在跳过用例，未完成全部数据库验证。", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


def run_smoke(check, pdf, timeout):
    codes = []
    if check in ("model", "all"):
        from scripts.smoke_model_tool_call import main as model_main
        codes.append(model_main(["--timeout", str(min(timeout, 45))]))
    if check in ("mineru", "all"):
        from scripts.smoke_mineru import main as mineru_main
        codes.append(mineru_main([str(pdf)]))
    return 1 if any(codes) else 0


def worker(args):
    if args.suite == "smoke":
        if args.list:
            print("model\nmineru")
            return 0
        return run_smoke(args.check, args.pdf, args.timeout)
    if args.suite == "unit":
        with offline_network_guard() as attempts:
            code = run_unittest("unit", listing=args.list, verbose=args.verbose, failfast=args.failfast)
        if attempts:
            print("离线测试发生被拦截的网络访问：" + ", ".join(attempts), file=sys.stderr)
            return 1
        return code
    return run_unittest("integration", listing=args.list, verbose=args.verbose, failfast=args.failfast)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("unit", "integration", "smoke"), default="unit", nargs="?")
    parser.add_argument("--env-file", type=Path, help="显式配置文件；在线 smoke 默认使用项目 .env")
    parser.add_argument("--check", choices=("model", "mineru", "all"), default="model", help="smoke 检查目标，会请求真实服务")
    parser.add_argument("--pdf", type=Path, help="MinerU smoke 要发送到服务端的本地 PDF")
    parser.add_argument("--timeout", type=float, default=180, help="整个测试子进程的最大秒数，默认 180")
    parser.add_argument("--list", action="store_true", help="仅列出用例，不执行测试或调用在线服务")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--failfast", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.suite != "smoke" and (args.pdf is not None or args.check != "model"):
        parser.error("--pdf / --check 仅用于 smoke")
    if args.suite == "smoke" and args.check in ("mineru", "all") and not args.list:
        if args.pdf is None or not args.pdf.expanduser().is_file() or args.pdf.suffix.lower() != ".pdf":
            parser.error("MinerU smoke 需要通过 --pdf 指定一个存在的 PDF 文件")
        args.pdf = args.pdf.expanduser().resolve()
    if args._worker:
        return worker(args)
    env_file = args.env_file
    if args.suite == "smoke" and env_file is None and not args.list:
        env_file = ROOT / ".env"
    try:
        env = build_environment(args.suite, env_file=env_file)
        validate_environment(args.suite, env, args.check, args.list)
    except ConfigurationError as exc:
        parser.error(str(exc))
    command = [sys.executable, str(Path(__file__).resolve()), args.suite, "--_worker", "--timeout", str(args.timeout)]
    for flag in ("list", "verbose", "failfast"):
        if getattr(args, flag):
            command.append("--" + flag)
    if args.suite == "smoke":
        command += ["--check", args.check]
        if args.pdf is not None:
            command += ["--pdf", str(args.pdf)]
    proc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        print("测试进程超时，已停止。", file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait()
        return 130


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
