import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import mlx.core as mx
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.qwen4_exp import StreamingExperts
from runtime.tests.test_qwen_resident_block import fixture
from research.qwen_resident_block import BoundedArenaPool
from research.qwen_ready_groups import ready_experts


class ReadyGroupsTests(unittest.TestCase):
    def test_resident_compute_starts_before_missing_finishes_and_preserves_lfu(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with ExpertCache(model,slots=12) as control,ExpertCache(model,slots=12) as candidate:
                candidate._pool=BoundedArenaPool(model,12)
                for c in (control,candidate):c.get_many(0,[0,1])
                value=mx.ones((1,1,2560),mx.bfloat16);indices=mx.array([[[3,0,2,1]]])
                expected=StreamingExperts(0,control)(value,indices);mx.eval(expected)
                event=threading.Event();original_read=candidate._read_expert_into_slot
                from research import qwen_ready_groups as ready
                original_project=ready.direct_project
                def read(layer,expert,slot):
                    if not event.wait(5):raise RuntimeError('GPU resident work was not submitted')
                    return original_read(layer,expert,slot)
                def project(*args):
                    result=original_project(*args);mx.async_eval(result);event.set();return result
                state=dict(calls=0,groups=0,max_groups=0)
                with patch.object(candidate,'_read_expert_into_slot',read),patch.object(ready,'direct_project',project):
                    actual=ready_experts(value,indices,candidate,0,mx.zeros_like(value),state)
                self.assertTrue(mx.array_equal(expected,actual).item());self.assertIn(state['groups'],(2,3))
                first_groups=state['groups']
                self.assertFalse(candidate._pinned_expert_keys)
                self.assertEqual(candidate.metrics.bytes_read,control.metrics.bytes_read)
                a={key:(e.slot,e.frequency,e.last_access) for key,e in control._entries.items()}
                b={key:(e.slot,e.frequency,e.last_access) for key,e in candidate._entries.items()}
                self.assertEqual(a,b)
                actual=ready_experts(value,indices,candidate,0,mx.zeros_like(value),state)
                self.assertTrue(mx.array_equal(expected,actual).item());self.assertEqual(state['groups'],first_groups+1)

    def test_read_failure_releases_new_slots_and_retry_recovers(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with ExpertCache(model,slots=12) as c:
                c._pool=BoundedArenaPool(model,12);c.get_many(0,[0])
                value=mx.ones((1,1,2560),mx.bfloat16);indices=mx.array([[[0,1,2,3]]]);state=dict(calls=0,groups=0,max_groups=0)
                with patch.object(c,'_read_expert_into_slot',side_effect=EOFError('truncated')):
                    with self.assertRaises(EOFError):ready_experts(value,indices,c,0,mx.zeros_like(value),state)
                self.assertFalse(c._pinned_expert_keys);self.assertEqual(set(c._entries),{(0,0)})
                self.assertEqual(len(c._free_slots),11)
                actual=ready_experts(value,indices,c,0,mx.zeros_like(value),state)
                expected=StreamingExperts(0,c)(value,indices);mx.eval(expected)
                self.assertTrue(mx.array_equal(actual,expected).item())

    def test_cancel_after_resident_submission_drains_before_slot_release(self):
        from research import qwen_ready_groups as ready
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with ExpertCache(model,slots=12) as c:
                c._pool=BoundedArenaPool(model,12);c.get_many(0,[0])
                event=threading.Event();original_read=c._read_expert_into_slot
                original_project=ready.direct_project
                def reader(*args):
                    if not event.wait(5):raise RuntimeError('missing resident submission')
                    return original_read(*args)
                def project(*args):
                    value=original_project(*args);mx.async_eval(value);event.set();return value
                value=mx.ones((1,1,2560),mx.bfloat16);indices=mx.array([[[0,1,2,3]]])
                with patch.object(c,'_read_expert_into_slot',reader), \
                     patch.object(ready,'direct_project',project), \
                     patch.object(ready,'check_cancelled',side_effect=[None,None,RuntimeError('cancel')]):
                    with self.assertRaisesRegex(RuntimeError,'cancel'):
                        ready_experts(value,indices,c,0,mx.zeros_like(value),dict(calls=0,groups=0,max_groups=0))
                self.assertTrue(event.is_set());self.assertFalse(c._pinned_expert_keys)
                self.assertEqual(set(c._entries),{(0,0)});self.assertEqual(len(c._free_slots),11)


if __name__=='__main__':unittest.main()
