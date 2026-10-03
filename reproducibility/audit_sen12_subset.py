"""Recheck extracted Sen12 files, identities and spatial overlap; no model inference."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .prepare_sen12_subset import inspect_sample, hash_file, write_json, write_csv


def overlap_pairs(rows):
    """Record positive-area intersections in each projected coordinate system."""
    result = []
    for i, left in enumerate(rows):
        for right in rows[i+1:]:
            if left['crs'] != right['crs']:
                continue
            width = min(left['xmax'], right['xmax']) - max(left['xmin'], right['xmin'])
            height = min(left['ymax'], right['ymax']) - max(left['ymin'], right['ymin'])
            if width > 1e-7 and height > 1e-7:
                result.append({'sample_a': left['sample'], 'sample_b': right['sample'],
                               'intersection_coordinate_area': width*height,
                               'official_split_a': left['official_split'],
                               'official_split_b': right['official_split']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source, output = Path(args.prepared), Path(args.out)
    if output.exists():
        raise FileExistsError('Use a fresh audit output directory')
    output.mkdir(parents=True)
    initial = json.loads((source / 'summary.json').read_text(encoding='utf8'))
    candidates = json.loads((source / 'official_candidates.json').read_text(encoding='utf8'))
    with (source / 'sample_manifest.csv').open(encoding='utf8', newline='') as stream:
        extraction = list(csv.DictReader(stream))
    started = time.monotonic()
    rows, failures, temporal = [], [], []
    notes, variable_extrema = Counter(), {}
    mask_mismatch, scl_unknown_frames, scl_unknown_samples = [], 0, 0
    groups = defaultdict(list)
    with (output / 'sample_details.jsonl').open('w', encoding='utf8') as details_stream:
        for position, previous in enumerate(extraction, 1):
            sample = previous['sample']
            if Path(sample).name != sample:
                raise ValueError('Invalid filename in extraction manifest')
            path = source / 'raw' / sample
            payload = path.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            if sha != previous['sha256'] or len(payload) != int(previous['bytes']):
                raise ValueError(f'Extracted file changed: {sample}')
            info = inspect_sample(payload)
            info.update(sample=sample, sha256=sha, archive=previous['archive'])
            details_stream.write(json.dumps(info, ensure_ascii=False, allow_nan=False) + '\n')
            if info['errors']:
                failures.append({'sample': sample, 'errors': info['errors']})
            notes.update(info['notes'])
            bbox = info['bbox']
            candidate = candidates.get(sample, {})
            positive = max(info['positive_pixels_by_time'])
            if candidate and positive != candidate['official_positive_pixels']:
                mask_mismatch.append({'sample': sample, 'measured_max': positive,
                                      'official': candidate['official_positive_pixels']})
            scl_unknown_samples += any(x > 0 for x in info['scl_255_fraction_by_time'])
            scl_unknown_frames += sum(x > 0 for x in info['scl_255_fraction_by_time'])
            for variable, values in info['variable_stats'].items():
                extrema = variable_extrema.setdefault(variable, {'min': values['min'], 'max': values['max'], 'nonfinite': 0})
                extrema['min'] = min(extrema['min'], values['min'])
                extrema['max'] = max(extrema['max'], values['max'])
                extrema['nonfinite'] += values['nonfinite']
            groups[info['spectral_array_sha256']].append(sample)
            row = {'sample': sample, 'archive': previous['archive'], 'bytes': len(payload), 'sha256': sha,
                   'schema_valid': not info['errors'], 'official_split': candidate.get('official_split', ''),
                   'official_positive_pixels': candidate.get('official_positive_pixels', ''),
                   'positive_pixels': positive, 'annotated': info['attributes']['annotated'],
                   'event_date': info['attributes']['event_date'], 'crs': info['attributes']['crs'],
                   'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3],
                   'mask_constant_across_time': info['mask_constant_across_time'],
                   'spectral_array_sha256': info['spectral_array_sha256'], 'notes': ';'.join(info['notes'])}
            rows.append(row)
            event = info['attributes']['event_date']
            if event and 'pre_post_indices' in info:
                indices = info['pre_post_indices']
                pre, post = info['dates'][indices['pre']][:10], info['dates'][indices['post']][:10]
                temporal.append({'sample': sample, 'event_date': event, 'pre_index': indices['pre'],
                                 'post_index': indices['post'], 'pre_date': pre, 'post_date': post,
                                 'pre_before_event': pre < event, 'post_on_or_after_event': post >= event,
                                 'post_same_day_as_event': post == event,
                                 'post_scl_255_fraction': info['scl_255_fraction_by_time'][indices['post']]})
            if position % 200 == 0:
                print(json.dumps({'rechecked': position, 'schema_errors': len(failures),
                                  'seconds': round(time.monotonic()-started, 1)}), flush=True)
    write_csv(output / 'sample_manifest.csv', rows, list(rows[0]))
    write_csv(output / 'temporal_metadata.csv', temporal, list(temporal[0]))
    overlaps = overlap_pairs(rows)
    write_csv(output / 'overlap_pairs.csv', overlaps,
              ['sample_a', 'sample_b', 'intersection_coordinate_area', 'official_split_a', 'official_split_b'])
    duplicates = [value for value in groups.values() if len(value) > 1]
    write_json(output / 'duplicate_spectral_arrays.json', duplicates)
    write_json(output / 'validation_errors.json', failures)
    write_json(output / 'official_mask_count_mismatches.json', mask_mismatch)
    actual = {row['sample'] for row in rows}
    missing = sorted(set(candidates)-actual)
    write_csv(output / 'missing_official_samples.csv', [{'sample': name} for name in missing], ['sample'])
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_revision': initial['dataset_revision'], 'archives_verified': initial['archives_verified'],
              'retained_samples': len(rows), 'retained_bytes': sum(row['bytes'] for row in rows),
              'by_archive': initial['by_archive'], 'files_match_extraction_sha256': True,
              'schema_invalid_samples': len(failures),
              'positive_mask_samples': sum(row['positive_pixels'] > 0 for row in rows),
              'empty_mask_samples': sum(row['positive_pixels'] == 0 for row in rows),
              'official_ld_candidates': len(candidates), 'missing_official_candidates': len(missing),
              'official_mask_count_mismatches': len(mask_mismatch),
              'official_split_counts': dict(Counter(row['official_split'] for row in rows if row['official_split'])),
              'exact_spectral_duplicate_groups': len(duplicates), 'positive_area_overlap_pairs': len(overlaps),
              'notes': dict(notes), 'variable_extrema': variable_extrema,
              'scl_255_samples': scl_unknown_samples, 'scl_255_frames': scl_unknown_frames,
              'scl_255_policy': 'Unknown quality code retained; not corruption or a standard SCL class. No training-frame selection made.',
              'post_metadata_before_event': sum(not row['post_on_or_after_event'] for row in temporal),
              'post_metadata_same_day_as_event': sum(row['post_same_day_as_event'] for row in temporal),
              'post_metadata_with_scl_255': sum(row['post_scl_255_fraction'] > 0 for row in temporal),
              'all_time_masks_constant': all(row['mask_constant_across_time'] for row in rows),
              'historical_40_10_210_split_recovered': False,
              'source_code_sha256': {p.name: hash_file(p) for p in
                                      [Path(__file__), Path(__file__).with_name('prepare_sen12_subset.py')]},
              'seconds': time.monotonic()-started}
    write_json(output / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
