"""统一测试入口的配置隔离、分组和失败退出码。"""

from contextlib import redirect_stderr
import io
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_tests as runner


class TestEnvironmentTests(unittest.TestCase):
    def test_unit_replaces_inherited_business_config_and_disables_dotenv(self):
        inherited = {
            "PATH": "/test/bin", "OPENAI_API_KEY": "business-key",
            "NEO4J_KG_TEST_URI": "bolt://business-db:7687",
            "MONGO_URL": "mongodb://business-db:27017", "BGE_DEVICE": "mps",
            "PROJECT_ROOT": "/some-other-checkout",
        }
        with patch.object(runner, "_read_env") as read:
            env = runner.build_environment("unit", environ=inherited)
        read.assert_not_called()
        self.assertEqual(inherited["OPENAI_API_KEY"], "business-key")
        self.assertEqual(env["PATH"], "/test/bin")
        self.assertEqual(env["OPENAI_API_KEY"], "unit-test-key")
        self.assertEqual(env["BGE_DEVICE"], "cpu")
        self.assertEqual(env["PROJECT_ROOT"], str(runner.ROOT))
        self.assertNotIn("NEO4J_KG_TEST_URI", env)
        self.assertEqual(env["PYTHON_DOTENV_DISABLED"], "1")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_unit_refuses_env_file(self):
        with self.assertRaises(runner.ConfigurationError):
            runner.build_environment("unit", environ={}, env_file=Path(".env"))

    def test_integration_does_not_fall_back_to_business_addresses(self):
        source = {"NEO4J_URI": "bolt://business-db:7687", "NEO4J_PASSWORD": "business-password", "MILVUS_URL": "http://business-db:19530"}
        env = runner.build_environment("integration", environ=source)
        self.assertTrue(all(not env.get(key) for key in runner.TEST_KEYS))
        with self.assertRaisesRegex(runner.ConfigurationError, "NEO4J_KG_TEST_URI"):
            runner.validate_environment("integration", env)

    def test_explicit_integration_file_only_imports_test_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.env"
            path.write_text("NEO4J_KG_TEST_URI=bolt://127.0.0.1:17687\nNEO4J_KG_TEST_PASSWORD=test-password\nMILVUS_KG_TEST_URI=http://127.0.0.1:29530\nOPENAI_API_KEY=should-not-load\n", encoding="utf-8")
            env = runner.build_environment("integration", environ={}, env_file=path)
        runner.validate_environment("integration", env)
        self.assertEqual(env["NEO4J_KG_TEST_URI"], "bolt://127.0.0.1:17687")
        self.assertEqual(env["OPENAI_API_KEY"], "unit-test-key")

    def test_smoke_requires_real_config_and_only_validates_requested_check(self):
        env = runner.build_environment("smoke", environ={})
        with self.assertRaisesRegex(runner.ConfigurationError, "OPENAI_API_KEY"):
            runner.validate_environment("smoke", env)
        runner.validate_environment("smoke", {"MINERU_API_TOKEN": "test", "MINERU_BASE_URL": "https://example.invalid"}, "mineru")


class TestSelectionTests(unittest.TestCase):
    def test_unit_and_integration_cases_do_not_mix(self):
        class UnitCase(unittest.TestCase):
            def runTest(self): pass

        class DatabaseCase(unittest.TestCase):
            test_scope = "integration"
            def runTest(self): pass

        unit, integration = UnitCase(), DatabaseCase()
        source = unittest.TestSuite([unittest.TestSuite([unit]), integration])
        self.assertEqual(list(runner.select_suite(source, "unit")), [unit])
        self.assertEqual(list(runner.select_suite(source, "integration")), [integration])

    def test_empty_test_suite_is_an_error(self):
        with patch.object(unittest.TestLoader, "discover", return_value=unittest.TestSuite()), redirect_stderr(io.StringIO()):
            self.assertEqual(runner.run_unittest("unit"), 2)

    def test_swallowed_network_error_still_fails_unit_entry(self):
        def accidentally_connect(*args, **kwargs):
            with socket.socket() as connection:
                try:
                    connection.connect(("127.0.0.1", 1))
                except runner.NetworkAccessBlocked:
                    pass
            return 0

        args = SimpleNamespace(suite="unit", list=False, verbose=False, failfast=False)
        with patch.object(runner, "run_unittest", side_effect=accidentally_connect), redirect_stderr(io.StringIO()):
            self.assertEqual(runner.worker(args), 1)

    def test_import_demo_does_not_import_the_live_graph(self):
        code = "import sys;sys.modules['app.import_process.agent.main_graph']=None; import app.test.test_import_main_graph; print('safe')"
        result = subprocess.run([sys.executable, "-c", code], cwd=runner.ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "safe")


class TestCommandTests(unittest.TestCase):
    def test_missing_integration_config_fails_before_launching_worker(self):
        with patch.dict(runner.os.environ, {}, clear=True), patch.object(runner.subprocess, "Popen") as launch, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                runner.main(["integration"])
        self.assertEqual(raised.exception.code, 2)
        launch.assert_not_called()

    def test_invalid_timeout_is_rejected(self):
        for value in ["0", "-1", "nan", "inf"]:
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(["--timeout", value])
                self.assertEqual(raised.exception.code, 2)

    def test_smoke_failures_produce_nonzero_exit(self):
        with patch("scripts.smoke_model_tool_call.main", return_value=0) as model, patch("scripts.smoke_mineru.main", return_value=1) as mineru:
            self.assertEqual(runner.run_smoke("all", Path("fixture.pdf"), 180), 1)
        model.assert_called_once_with(["--timeout", "45"])
        mineru.assert_called_once_with(["fixture.pdf"])


if __name__ == "__main__":
    unittest.main()
