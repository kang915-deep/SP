"""CPU-only mocked regression; this does not test real GPUs or four-card training."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CudaMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[1] / "train.py"
        cls.tree = ast.parse(cls.source.read_text(encoding="utf-8"))
        assignments = [node for node in ast.walk(cls.tree)
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Attribute) and target.attr == "available_memory"
                           for target in node.targets)]
        if len(assignments) != 1:
            raise AssertionError("Expected one real available_memory override in train.py")
        cls.override = assignments[0]

    def test_override_order(self):
        lines = {name: [node.lineno for node in ast.walk(self.tree)
                        if isinstance(node, ast.Call) and ast.unparse(node.func) == name]
                 for name in ("torch.cuda.set_device", "dist.init_process_group", "deepspeed.initialize")}
        self.assertTrue(all(len(value) == 1 for value in lines.values()))
        self.assertLess(lines["torch.cuda.set_device"][0], self.override.lineno)
        self.assertLess(self.override.lineno, lines["dist.init_process_group"][0])
        self.assertLess(self.override.lineno, lines["deepspeed.initialize"][0])
        self.assertIn(self.override, self.tree.body)

    def test_uuid_mapping_and_driver_free_memory_mock(self):
        # Logical ranks deliberately differ from the underlying UUID ordering.
        uuids = [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in range(4)]
        visible = [uuids[3], uuids[0], uuids[2], uuids[1]]
        free = {uuid: gib * 2**30 for uuid, gib in zip(uuids, (29, 52, 12, 37))}
        total, allocated = 80 * 2**30, 2 * 2**30
        nvml = Mock(side_effect=AssertionError("NVML must not be called"))

        def old_available_memory(device_index=None):
            # Reproduce the failing DeepSpeed UUID-to-integer parsing branch.
            indices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            return nvml(indices[0 if device_index is None else device_index])

        current = 0

        def mem_get_info(device_index=None):
            if device_index is None:
                logical = current
            elif isinstance(device_index, int):
                logical = device_index
            elif isinstance(device_index, str):
                logical = current if device_index == "cuda" else int(device_index.split(":")[1])
            else:
                self.assertEqual(device_index.type, "cuda")
                logical = current if device_index.index is None else device_index.index
            return free[visible[logical]], total

        cuda = SimpleNamespace(mem_get_info=Mock(side_effect=mem_get_info))
        accelerator = SimpleNamespace(
            available_memory=old_available_memory,
            _get_nvml_gpu_id=nvml,
            total_memory=Mock(side_effect=AssertionError("Do not estimate free memory from total")),
            memory_allocated=Mock(side_effect=AssertionError("Do not subtract only this process's allocation")),
        )
        scope = {"torch": SimpleNamespace(cuda=cuda),
                 "deepspeed": SimpleNamespace(accelerator=SimpleNamespace(get_accelerator=lambda: accelerator)),
                 "pynvml": SimpleNamespace(nvmlDeviceGetMemoryInfo=nvml)}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ",".join(visible)}):
            environment = dict(os.environ)
            with self.assertRaises(ValueError):
                accelerator.available_memory(0)
            # Execute only the actual production assignment, never the train entry point.
            code = compile(ast.Module(body=[self.override], type_ignores=[]), str(self.source), "exec")
            exec(code, scope)
            for current in range(4):
                expected = free[visible[current]]
                self.assertNotEqual(expected, total - allocated)
                self.assertEqual(accelerator.available_memory(), expected)
                for argument in (None, "cuda", SimpleNamespace(type="cuda", index=None)):
                    self.assertEqual(accelerator.available_memory(argument), expected)
                    self.assertIs(cuda.mem_get_info.call_args.args[0], argument)
                for rank in range(4):
                    for argument in (rank, f"cuda:{rank}", SimpleNamespace(type="cuda", index=rank)):
                        self.assertEqual(accelerator.available_memory(device_index=argument), free[visible[rank]])
                        self.assertIs(cuda.mem_get_info.call_args.args[0], argument)
            self.assertEqual(dict(os.environ), environment, "Production override changed the UUID mapping")
        nvml.assert_not_called()
        accelerator.total_memory.assert_not_called()
        accelerator.memory_allocated.assert_not_called()


if __name__ == "__main__":
    print("CPU MOCK regression only; no real CUDA/NVML or four-GPU execution.", flush=True)
    unittest.main()
