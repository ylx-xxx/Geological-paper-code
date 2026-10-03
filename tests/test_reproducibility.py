import unittest
import numpy as np
from reproducibility.core import CHANNELS, normalize, confusion, metrics, verify_selection_split
from reproducibility.audit_dataset import cross_split_duplicates

class ScientificInvariants(unittest.TestCase):
    def test_rgb_is_b4_b3_b2(self):
        self.assertEqual(CHANNELS['rgb'],[3,2,1])
        self.assertEqual(CHANNELS['first3_legacy'],[0,1,2])
    def test_normalization_matches_legacy(self):
        from train_l4s_qz.dataset import Landslide4SenseDataset
        x=np.random.default_rng(42).normal(size=(128,128,14)).astype(np.float32)
        x[0,0,0]=np.nan;x[1,1,1]=np.inf;x[:,:,13]=5
        expected=Landslide4SenseDataset._normalize_14band(None,x)
        np.testing.assert_array_equal(normalize(x),expected)
        np.testing.assert_array_equal(normalize(x.transpose(2,0,1)),expected)
    def test_global_is_pooled_not_patch_mean(self):
        a=confusion(np.array([1,0]),np.array([1,1]))
        b=confusion(np.zeros(100,dtype=int),np.ones(100,dtype=int))
        self.assertAlmostEqual(metrics(a+b)['Landslide_IoU'],1/102)
        self.assertNotEqual(metrics(a+b)['Landslide_IoU'],.25)
    def test_final_counts_reproduce_scores(self):
        m=metrics([[12765924,93745],[83950,163581]])
        self.assertAlmostEqual(m['Landslide_IoU'],.47932172200799356)
        self.assertAlmostEqual(m['Landslide_F1'],.6480290458486264)
    def test_ignore_and_invalid_labels(self):
        self.assertEqual(confusion(np.array([1,1]),np.array([1,255])).sum(),1)
        with self.assertRaises(ValueError):confusion(np.array([1]),np.array([2]))
    def test_test_cannot_select_checkpoint(self):
        verify_selection_split('val')
        with self.assertRaises(ValueError):verify_selection_split('test')
    def test_empty_foreground_is_not_perfect_iou(self):
        self.assertEqual(metrics([[100,0],[0,0]])['Landslide_IoU'],0.)
    def test_duplicates_use_content_not_sample_names(self):
        rows=[{'split':'train','sample':'image_1','array_sha256':'same'},
              {'split':'test','sample':'image_500','array_sha256':'same'},
              {'split':'test','sample':'image_1','array_sha256':'different'}]
        self.assertEqual(len(cross_split_duplicates(rows)),1)
    def test_relative_bias_resize_preserves_constant_heads(self):
        import torch
        from reproducibility.train import resize_bias
        source=torch.ones(169,3)*torch.tensor([1.,2.,3.])
        actual=resize_bias(source,torch.zeros(49,3))
        torch.testing.assert_close(actual,torch.ones(49,3)*torch.tensor([1.,2.,3.]))
        with self.assertRaises(ValueError):resize_bias(source,torch.zeros(49,4))

if __name__=='__main__':unittest.main()
