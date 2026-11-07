"""Execute the real argparse, protocol and DeepSpeed config AST without GPUs."""
import argparse
import ast
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class NoOverlapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "train.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assignments = {ast.unparse(node.targets[0]): i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)}
        cuda = next(i for i, node in enumerate(tree.body)
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "torch.cuda.set_device")
        cls.entry = compile(ast.Module(body=tree.body[assignments["parser"]:cuda], type_ignores=[]), str(source), "exec")
        cls.protocol = compile(ast.Module(body=tree.body[assignments["protocol"]:assignments["out"]], type_ignores=[]), str(source), "exec")
        cls.config = compile(ast.Module(body=[tree.body[assignments["config"]]], type_ignores=[]), str(source), "exec")

    def settings(self, flags):
        argv = ["train.py", "--qwen", "q", "--wan", "w", "--data", "d", "--output", "o", *flags]
        env = {"argparse": argparse, "Path": Path, "os": os, "hashlib": hashlib, "raw": b"train"}
        with patch.object(sys, "argv", argv), patch.dict(os.environ, RANK="0", WORLD_SIZE="4", LOCAL_RANK="0", CUDA_VISIBLE_DEVICES="0,1,2,3"), tempfile.TemporaryDirectory() as directory:
            env["data_dir"] = Path(directory)
            (env["data_dir"] / "tune.json").write_bytes(b"tune")
            exec(self.entry, env)
            exec(self.protocol, env)
            exec(self.config, env)
        return env["args"], env["protocol"], env["config"]

    def test_default_retains_exact_existing_protocol_and_config(self):
        args, protocol, config = self.settings([])
        self.assertFalse(args.no_overlap)
        self.assertEqual(protocol, {
            "train_sha256": hashlib.sha256(b"train").hexdigest(),
            "tune_sha256": hashlib.sha256(b"tune").hexdigest(),
            "world": 4, "micro_batch": 1, "effective_batch": 64, "smoke": False, "cpu_adam": False,
        })
        self.assertEqual(config, {
            "train_batch_size": 64, "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 16,
            "gradient_clipping": 1., "bf16": {"enabled": True}, "zero_allow_untested_optimizer": True,
            "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                                  "reduce_bucket_size": 50000000, "allgather_bucket_size": 50000000},
            "steps_per_print": 10000,
        })

    def test_flag_changes_only_overlap_and_records_resume_identity(self):
        for flags in ([], ["--diagnose-numerics", "--benchmark-steps", "2", "--diagnostic-micros", "2"]):
            with self.subTest(flags=flags):
                _, default_protocol, default_config = self.settings(flags)
                args, protocol, config = self.settings([*flags, "--no-overlap"])
                self.assertTrue(args.no_overlap)
                expected = copy.deepcopy(default_config)
                expected["zero_optimization"]["overlap_comm"] = False
                self.assertEqual(config, expected)
                self.assertEqual(protocol, dict(default_protocol, overlap_comm=False))
                self.assertNotEqual(protocol, default_protocol)
        _, fresh_protocol, _ = self.settings(["--no-overlap"])
        _, resumed_protocol, _ = self.settings(["--no-overlap", "--resume", "step-000250"])
        self.assertEqual(fresh_protocol, resumed_protocol)


if __name__ == "__main__":
    unittest.main()
