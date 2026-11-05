import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from media_resume import archive_priority,restored_coverage


class MediaResumeTests(unittest.TestCase):
    def test_prepare_repairs_missing_media_and_reuses_completed_extraction(self):
        import hashlib,io,runpy,sys,tarfile,types
        archive=io.BytesIO()
        payload=b'frame contents'*100
        with tarfile.open(fileobj=archive,mode='w:gz') as tar:
            info=tarfile.TarInfo('clip/0001.jpg');info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))
        raw=archive.getvalue()
        item={'name':'train_300k/chunk_12.tar.gz','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
              'repo':'ShareGPTVideo/train_video_and_instruction','revision':'test'}
        modules={'hound_archive_manifest':types.SimpleNamespace(ITEMS=[item]),
                 'vsi_media_archives':types.SimpleNamespace(VSI_ARCHIVES=[])}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);downloads=root/'downloads';downloads.mkdir()
            target=downloads/'chunk_12.tar.gz';target.write_bytes(raw)
            for split in ['train','tune','confirm']:
                (root/(split+'.json')).write_text(json.dumps([{'kind':'frames','media':'clip'}] if split=='train' else []))
            argv=['prepare_data.py','--output',str(root),'--stage','media']
            def run():
                with patch.dict(sys.modules,modules),patch.object(sys,'argv',argv):
                    runpy.run_path(str(Path(__file__).with_name('prepare_data.py')),run_name='__main__')
            run()
            media=root/'media/hound/clip/0001.jpg'
            self.assertEqual(media.read_bytes(),payload)
            with patch('tarfile.open',side_effect=AssertionError('Completed archive was scanned again')):
                run()
            media.unlink()
            run()
            self.assertEqual(media.read_bytes(),payload)
            self.assertEqual(json.loads((root/'media_ready.json').read_text()),{'video':0,'frames':1})

    def test_ranges_only_interruption_precedes_complete_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            items=[{'name':f'chunk_{i}.tar.gz','bytes':12} for i in range(3)]
            (root/'chunk_0.tar.gz').write_bytes(b'x'*12)
            folder=root/'chunk_1.tar.gz.part.ranges';folder.mkdir()
            (folder/'0-3').write_bytes(b'abcd')
            (folder/'4-7.incomplete.partial').write_bytes(b'x')
            ordered=sorted(items,key=lambda item:archive_priority(item,root))
            self.assertEqual([x['name'] for x in ordered],['chunk_1.tar.gz','chunk_2.tar.gz','chunk_0.tar.gz'])
            (folder/'4-11').write_bytes(b'e'*8)
            self.assertEqual(archive_priority(items[1],root),2)

    def test_receipt_reuse_requires_matching_selection_and_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);media=root/'clip.mp4';media.write_bytes(b'abcd')
            receipt=root/'receipt.json';identity={'archive_sha256':'archive','selection_sha256':'selected'}
            receipt.write_text(json.dumps(dict(identity=identity,files=[['clip.mp4',4]],covered=['clip.mp4'])))
            self.assertEqual(restored_coverage(receipt,root,identity),{'clip.mp4'})
            self.assertIsNone(restored_coverage(receipt,root,{**identity,'selection_sha256':'other'}))
            media.write_bytes(b'a')
            self.assertIsNone(restored_coverage(receipt,root,identity))


if __name__=='__main__':unittest.main()
