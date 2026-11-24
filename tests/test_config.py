import tomllib
import unittest
from pathlib import Path

from tests.helpers import IsolatedTestCase
from xarencode import config


class ConfigTests(IsolatedTestCase):
    def test_template_matches_defaults(self):
        parsed = tomllib.loads(config.TEMPLATE)
        merged = config._merge(config.DEFAULTS, parsed)
        self.assertEqual(merged, config.DEFAULTS)

    def test_first_run_writes_template(self):
        cfg = config.load_config()
        self.assertTrue(config.config_path().exists())
        self.assertEqual(cfg["server"]["port"], 8091)
        self.assertEqual(cfg["dev"], {})  # presets are commented out

    def test_user_override_merges(self):
        p = self.tmp / "c.toml"
        p.write_text('[server]\nport = 9000\n[dev.a]\ndir="/x"\ncmd="y"\n')
        cfg = config.load_config(p)
        self.assertEqual(cfg["server"]["port"], 9000)
        self.assertEqual(cfg["server"]["host"], "127.0.0.1")
        self.assertEqual(cfg["dev"]["a"]["cmd"], "y")

    def test_missing_explicit_config(self):
        with self.assertRaises(FileNotFoundError):
            config.load_config(self.tmp / "nope.toml")

    def test_resolve_model(self):
        cfg = config.default_config()
        self.assertEqual(config.resolve_model(cfg), config.data_dir() / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        cfg["model"]["gguf"] = "/abs/m.gguf"
        self.assertEqual(config.resolve_model(cfg), Path("/abs/m.gguf"))
        cfg["model"]["gguf"] = "bad"
        with self.assertRaises(ValueError):
            config.resolve_model(cfg)

    def test_short_name(self):
        self.assertEqual(config.model_short_name(config.default_config()), "Qwen3-4B")

    def test_paths_follow_xdg(self):
        self.assertEqual(config.state_dir(), self.tmp / "state" / "xarencode")


if __name__ == "__main__":
    unittest.main()
