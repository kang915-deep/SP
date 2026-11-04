"""Linux process-tree tests: stop only data preparation, never an active training stage."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(sys.platform == 'linux', 'Linux /proc and signals required')
class ResumeTests(unittest.TestCase):
    def run_case(self, stage):
        with tempfile.TemporaryDirectory(prefix='vega-resume-test-') as tmp:
            repo = Path(tmp)
            scripts = repo/'qwen3_vega/scripts'
            scripts.mkdir(parents=True)
            shutil.copyfile(Path(__file__).with_name('resume_download.py'), scripts/'resume_download.py')
            work = repo/'disk/qwen3-vega'
            work.mkdir(parents=True)
            (repo/'start.sh').write_text('''#!/bin/bash
if [[ $1 == --worker ]]; then
  python3 "$(dirname "$0")/qwen3_vega/scripts/one_click.py" "$2"
else
  echo restarted > "$(dirname "$0")/restarted"
fi
''')
            (scripts/'one_click.py').write_text(
                "import subprocess,sys\nfrom pathlib import Path\n"
                f"subprocess.run([sys.executable,str(Path(__file__).with_name('{stage}.py')),'--output',sys.argv[1]+'/data'])\n")
            (scripts/(stage+'.py')).write_text(
                "from pathlib import Path\nimport time\n"
                "Path(__file__).with_suffix('.started').touch()\ntime.sleep(90)\n")
            sentinel = subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)'])
            job = subprocess.Popen(['bash',str(repo/'start.sh'),'--worker',str(work)], start_new_session=True)
            try:
                deadline=time.monotonic()+5
                while not (scripts/(stage+'.started')).exists() and time.monotonic()<deadline:
                    time.sleep(.05)
                self.assertTrue((scripts/(stage+'.started')).exists())
                result=subprocess.run([sys.executable,str(scripts/'resume_download.py'),str(repo/'disk'),'0,1,2,3'],
                                      capture_output=True,text=True,timeout=15)
                self.assertIsNone(sentinel.poll())
                if stage=='prepare_data':
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertTrue((repo/'restarted').is_file())
                    job.wait(timeout=3)
                else:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('not in data preparation', result.stderr)
                    self.assertIsNone(job.poll())
                    self.assertFalse((repo/'restarted').exists())
            finally:
                sentinel.terminate();sentinel.wait()
                if job.poll() is None:
                    import signal
                    os.killpg(job.pid,signal.SIGTERM)
                    job.wait()

    def test_download_only_restart(self):
        self.run_case('prepare_data')

    def test_refuse_training(self):
        self.run_case('train')


if __name__=='__main__':
    unittest.main()
