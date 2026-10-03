"""Integrity and identity checks for regional Sen12 preparation."""
import tarfile
import unittest
from io import BytesIO

import h5py
import numpy as np

from reproducibility.prepare_sen12_subset import VARIABLES, dates_from_cf, inspect_sample, official_candidates, safe_member_name
from reproducibility.audit_sen12_subset import overlap_pairs


class Sen12Preparation(unittest.TestCase):
    def test_candidate_count_uses_paths_not_other_strings(self):
        splits = {'train': [{'id': 'chimanimani_7', 'inventory': 'chimanimani',
                             's2': 's2/chimanimani_s2_7.nc', 'pixel_annotated': 64}],
                  'val': [], 'test': []}
        result = official_candidates(splits)
        self.assertEqual(list(result), ['chimanimani_s2_7.nc'])
        self.assertEqual(result['chimanimani_s2_7.nc']['official_positive_pixels'], 64)

    def test_repeated_official_sample_across_splits_is_rejected(self):
        sample = {'id': 'chimanimani_7', 'inventory': 'chimanimani',
                  's2': 's2/chimanimani_s2_7.nc', 'pixel_annotated': 64}
        with self.assertRaises(ValueError):
            official_candidates({'train': [sample], 'val': [], 'test': [sample]})

    def test_unsafe_archive_paths_and_links_are_rejected(self):
        for name in ['../sample.nc', '/sample.nc', 'C:/sample.nc', 'a\\sample.nc']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member_name(tarfile.TarInfo(name))
        member = tarfile.TarInfo('chimanimani_s2_7.nc')
        member.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            safe_member_name(member)
        member.type = tarfile.REGTYPE
        self.assertEqual(safe_member_name(member), 'chimanimani_s2_7.nc')

    def test_cf_dates_cross_leap_day_and_preserve_fractional_days(self):
        self.assertEqual(dates_from_cf([0, 1, 2.5], 'days since 2020-02-28', 'proleptic_gregorian'),
                         ['2020-02-28T00:00:00', '2020-02-29T00:00:00', '2020-03-01T12:00:00'])
        with self.assertRaises(ValueError):
            dates_from_cf([1], 'days since 2020-01-01', '360_day')

    def test_scl_unknown_quality_does_not_imply_file_corruption(self):
        memory = BytesIO()
        with h5py.File(memory, 'w') as data:
            shape = (15, 128, 128)
            for name in VARIABLES:
                data.create_dataset(name, data=np.full(shape, 255 if name == 'SCL' else 0, dtype='int16'))
            data['x'] = np.arange(128, dtype=float)*10
            data['y'] = np.arange(128, dtype=float)*10
            data['time'] = np.arange(15)
            data['time'].attrs['units'] = 'days since 2020-01-01'
            data['spatial_ref'] = 0
            data.attrs['crs'] = 'EPSG:32736'
            data.attrs['satellite'] = 's2'
            data.attrs['annotated'] = 'False'
        result = inspect_sample(memory.getvalue())
        self.assertEqual(result['errors'], [])
        self.assertIn('scl_255_quality_unknown', result['notes'])
        self.assertEqual(result['scl_255_fraction_by_time'], [1.]*15)
        with h5py.File(memory, 'r+') as data:
            data['MASK'][0, 0, 0] = 2
        self.assertIn('nonbinary_mask', inspect_sample(memory.getvalue())['errors'])

    def test_touching_tiles_do_not_count_as_area_overlap(self):
        def tile(name, xmin, xmax):
            return {'sample':name, 'crs':'EPSG:32736', 'official_split':'train',
                    'xmin':xmin, 'xmax':xmax, 'ymin':0, 'ymax':10}
        rows = [tile('a',0,10), tile('b',10,20), tile('c',5,15)]
        pairs = overlap_pairs(rows)
        self.assertEqual({(r['sample_a'],r['sample_b']) for r in pairs}, {('a','c'),('b','c')})
        self.assertTrue(all(r['intersection_coordinate_area'] == 50 for r in pairs))


if __name__ == '__main__':
    unittest.main()
