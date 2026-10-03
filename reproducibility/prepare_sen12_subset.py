"""Verify local Sen12 archives and inventory a regional subset without selecting a model."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

import h5py
import numpy as np


BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
VARIABLES = BANDS + ['DEM', 'MASK', 'SCL']
REGION = 'chimanimani'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def attribute_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf8')
    array = np.asarray(value)
    if array.size == 1 and array.ndim:
        return attribute_text(array.item())
    return str(value)


def official_candidates(splits, region=REGION):
    """Read the modality path field only; IDs and inventory names are not files."""
    result = {}
    for split, entries in splits.items():
        if split not in {'train', 'val', 'test'} or not isinstance(entries, list):
            raise ValueError('Unexpected official split schema')
        for entry in entries:
            if entry.get('inventory') != region:
                continue
            name = PurePosixPath(entry['s2']).name
            if not re.fullmatch(rf'{re.escape(region)}_s2_\d+\.nc', name):
                raise ValueError(f'Unexpected regional filename: {name}')
            if name in result:
                raise ValueError(f'Duplicate official sample: {name}')
            result[name] = {'official_split': split, 'official_id': entry['id'],
                            'official_positive_pixels': int(entry['pixel_annotated'])}
    return result


def safe_member_name(member):
    path = PurePosixPath(member.name)
    if path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name:
        raise ValueError(f'Unsafe archive path: {member.name}')
    if not member.isfile() and not member.isdir():
        raise ValueError(f'Unsupported archive member type: {member.name}')
    return path.name


def dates_from_cf(values, units, calendar):
    match = re.fullmatch(r'(days|hours|seconds) since (.+)', units)
    if not match or calendar not in {'standard', 'gregorian', 'proleptic_gregorian'}:
        raise ValueError(f'Unsupported time encoding: {units}; {calendar}')
    origin = datetime.fromisoformat(match.group(2).replace('Z', '+00:00'))
    seconds = {'days': 86400, 'hours': 3600, 'seconds': 1}[match.group(1)]
    return [(origin + timedelta(seconds=float(v) * seconds)).isoformat() for v in values]


def inspect_sample(payload):
    details = {'errors': [], 'notes': []}
    with h5py.File(BytesIO(payload), 'r') as data:
        missing = sorted(set(VARIABLES + ['x', 'y', 'time', 'spatial_ref']) - set(data))
        if missing:
            details['errors'].append('missing_variables:' + ','.join(missing))
            return details
        shapes = {name: list(data[name].shape) for name in VARIABLES}
        details['shapes'] = shapes
        if any(shape != [15, 128, 128] for shape in shapes.values()):
            details['errors'].append('unexpected_variable_shape')
        if details['errors']:
            return details
        dimensions = [[attribute_text(v) for v in data['B04'].dims[i].keys()] for i in range(3)]
        details['rgb_dimensions'] = dimensions
        arrays = {name: data[name][:] for name in VARIABLES}
        stats = {}
        for name, array in arrays.items():
            stats[name] = {'dtype': str(array.dtype), 'min': float(np.nanmin(array)),
                           'max': float(np.nanmax(array)), 'nonfinite': int((~np.isfinite(array)).sum())}
            if stats[name]['nonfinite']:
                details['errors'].append('nonfinite:' + name)
        details['variable_stats'] = stats
        mask = arrays['MASK']
        labels = np.unique(mask)
        if not np.isin(labels, [0, 1]).all():
            details['errors'].append('nonbinary_mask')
        counts = (mask == 1).sum(axis=(1, 2)).tolist()
        details['positive_pixels_by_time'] = counts
        details['mask_constant_across_time'] = bool(np.all(mask == mask[0]))
        details['mask_sha256'] = hashlib.sha256(mask.tobytes()).hexdigest()
        image_hash = hashlib.sha256()
        for band in BANDS:
            image_hash.update(band.encode('ascii'))
            image_hash.update(arrays[band].tobytes())
        details['spectral_array_sha256'] = image_hash.hexdigest()
        attrs = {name: attribute_text(data.attrs.get(name, '')) for name in
                 ['annotated', 'event_date', 'pre_post_dates', 'date_confidence', 'crs', 'satellite']}
        details['attributes'] = attrs
        if attrs['satellite'] != 's2':
            details['errors'].append('unexpected_sensor')
        if not attrs['crs']:
            details['errors'].append('missing_crs')
        if attrs['annotated'].lower() not in {'true', 'false'}:
            details['errors'].append('unknown_annotation_status')
        if not attrs['event_date']:
            details['notes'].append('event_date_absent')
        x, y = data['x'][:], data['y'][:]
        if x.shape != (128,) or y.shape != (128,) or not np.isfinite(x).all() or not np.isfinite(y).all():
            details['errors'].append('invalid_coordinates')
        else:
            dx, dy = np.diff(x), np.diff(y)
            if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]) or not np.isclose(abs(dx[0]), 10) or not np.isclose(abs(dy[0]), 10):
                details['errors'].append('unexpected_spatial_spacing')
            details['bbox'] = [float(x.min() - abs(dx[0])/2), float(y.min() - abs(dy[0])/2),
                               float(x.max() + abs(dx[0])/2), float(y.max() + abs(dy[0])/2)]
        values = data['time'][:]
        if values.shape != (15,) or not np.all(np.diff(values) > 0):
            details['errors'].append('invalid_time_order')
        units = attribute_text(data['time'].attrs.get('units', ''))
        calendar = attribute_text(data['time'].attrs.get('calendar', 'standard'))
        dates = dates_from_cf(values, units, calendar)
        details['dates'] = dates
        details['time_encoding'] = {'units': units, 'calendar': calendar}
        if attrs['pre_post_dates']:
            mapping = ast.literal_eval(attrs['pre_post_dates'])
            details['pre_post_indices'] = mapping
            if not isinstance(mapping, dict) or not all(isinstance(v, int) and 0 <= v < 15 for v in mapping.values()):
                details['errors'].append('invalid_pre_post_indices')
        scl = arrays['SCL']
        # 255 occurs in the hash-verified upstream archives without a declared
        # CF _FillValue. Preserve it as unknown quality, not a land-cover class
        # or evidence of archive corruption. Downstream cloud policy is separate.
        if not np.isin(scl, list(range(12)) + [255]).all():
            details['errors'].append('unexpected_scl_labels')
        if np.any(scl == 255):
            details['notes'].append('scl_255_quality_unknown')
        details['scl_histograms'] = []
        for frame in scl:
            labels, counts = np.unique(frame, return_counts=True)
            details['scl_histograms'].append({str(int(k)): int(n) for k, n in zip(labels, counts)})
        details['scl_255_fraction_by_time'] = (scl == 255).mean(axis=(1, 2)).tolist()
        rgb = np.stack([arrays[name] for name in ['B04', 'B03', 'B02']])
        details['rgb_all_zero_fraction_by_time'] = np.all(rgb == 0, axis=0).mean(axis=(1, 2)).tolist()
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-dir', required=True)
    parser.add_argument('--official-manifest', required=True)
    parser.add_argument('--official-splits', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-free-gib', type=float, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    source = Path(args.archives_dir).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; existing data are never overwritten')
    output.mkdir(parents=True)
    raw = output / 'raw'
    raw.mkdir()
    official = json.loads(Path(args.official_manifest).read_text(encoding='utf8'))
    splits = json.loads(Path(args.official_splits).read_text(encoding='utf8'))
    candidates = official_candidates(splits)
    write_json(output / 'official_source.json', official)
    write_json(output / 'official_candidates.json', candidates)
    archives = []
    for entry in official['files']:
        path = source / PurePosixPath(entry['path']).name
        if path.exists():
            archives.append((path, entry))
    if not archives:
        raise FileNotFoundError('No official archives found')
    verification = []
    for path, entry in archives:
        print(json.dumps({'phase': 'hash_archive', 'file': path.name}), flush=True)
        actual = hash_file(path)
        record = {'archive': path.name, 'bytes': path.stat().st_size,
                  'sha256': actual, 'expected_sha256': entry['lfs']['oid'],
                  'verified': actual == entry['lfs']['oid'] and path.stat().st_size == entry['size']}
        verification.append(record)
        write_json(output / 'archive_verification.json', verification)
        if not record['verified']:
            raise ValueError(f'Archive integrity mismatch: {path.name}')
        print(json.dumps(record), flush=True)
    rows, members, errors = [], [], []
    counts_by_archive = Counter()
    seen = set()
    manifest_fields = ['sample', 'archive', 'bytes', 'sha256', 'schema_valid', 'official_split',
                       'official_positive_pixels', 'annotated', 'event_date', 'crs', 'xmin', 'ymin',
                       'xmax', 'ymax', 'positive_pixels_max', 'mask_constant_across_time',
                       'spectral_array_sha256', 'mask_sha256', 'errors', 'notes']
    with (output / 'sample_details.jsonl').open('w', encoding='utf8') as detail_stream, \
         (output / 'sample_manifest.csv').open('w', encoding='utf8', newline='') as manifest_stream:
        writer = csv.DictWriter(manifest_stream, fieldnames=manifest_fields)
        writer.writeheader()
        for path, entry in archives:
            print(json.dumps({'phase': 'extract_and_inspect', 'file': path.name}), flush=True)
            with tarfile.open(path, mode='r|gz') as archive:
                for member in archive:
                    name = safe_member_name(member)
                    selected = member.isfile() and re.fullmatch(r'chimanimani_s2_\d+\.nc', name) is not None
                    members.append({'archive': path.name, 'member': member.name, 'bytes': member.size,
                                    'is_file': member.isfile(), 'retained': selected})
                    if not selected:
                        continue
                    if name in seen:
                        raise ValueError(f'Duplicate filename across archives: {name}')
                    seen.add(name)
                    if not 0 < member.size < 128 * 1024 * 1024:
                        raise ValueError(f'Unexpected member size: {name}')
                    if shutil.disk_usage(raw).free < member.size + args.min_free_gib * 1024**3:
                        raise RuntimeError('Free-space threshold reached')
                    stream = archive.extractfile(member)
                    payload = stream.read(member.size + 1)
                    if len(payload) != member.size:
                        raise ValueError(f'Incomplete member: {name}')
                    target = raw / name
                    if target.resolve().parent != raw.resolve():
                        raise ValueError('Extraction target escaped output directory')
                    with target.open('xb') as destination:
                        destination.write(payload)
                    sha = hashlib.sha256(payload).hexdigest()
                    try:
                        details = inspect_sample(payload)
                    except Exception as error:
                        details = {'errors': [f'{type(error).__name__}:{error}'], 'notes': []}
                    details.update(sample=name, archive=path.name, sha256=sha)
                    detail_stream.write(json.dumps(details, ensure_ascii=False, allow_nan=False) + '\n')
                    bbox = details.get('bbox', [None] * 4)
                    attr = details.get('attributes', {})
                    row = {'sample': name, 'archive': path.name, 'bytes': member.size, 'sha256': sha,
                           'schema_valid': not details['errors'],
                           'official_split': candidates.get(name, {}).get('official_split', ''),
                           'official_positive_pixels': candidates.get(name, {}).get('official_positive_pixels', ''),
                           'annotated': attr.get('annotated', ''), 'event_date': attr.get('event_date', ''),
                           'crs': attr.get('crs', ''), 'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3],
                           'positive_pixels_max': max(details.get('positive_pixels_by_time', [0])),
                           'mask_constant_across_time': details.get('mask_constant_across_time', ''),
                           'spectral_array_sha256': details.get('spectral_array_sha256', ''),
                           'mask_sha256': details.get('mask_sha256', ''),
                           'errors': ';'.join(details['errors']), 'notes': ';'.join(details['notes'])}
                    writer.writerow(row)
                    rows.append(row)
                    counts_by_archive[path.name] += 1
                    if details['errors']:
                        errors.append({'sample': name, 'errors': details['errors']})
                    if len(rows) % 100 == 0:
                        manifest_stream.flush()
                        detail_stream.flush()
                        print(json.dumps({'inspected_samples': len(rows), 'invalid_samples': len(errors),
                                          'seconds': round(time.monotonic() - started, 1)}), flush=True)
    write_csv(output / 'archive_members.csv', members, ['archive', 'member', 'bytes', 'is_file', 'retained'])
    missing = sorted(set(candidates) - seen)
    write_csv(output / 'missing_official_samples.csv', [{'sample': name} for name in missing], ['sample'])
    groups = defaultdict(list)
    for row in rows:
        if row['spectral_array_sha256']:
            groups[row['spectral_array_sha256']].append(row['sample'])
    duplicates = [value for value in groups.values() if len(value) > 1]
    write_json(output / 'duplicate_spectral_arrays.json', duplicates)
    write_json(output / 'validation_errors.json', errors)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_revision': official['revision'], 'archives_verified': len(archives),
              'archives_preserved': True, 'retained_samples': len(rows),
              'retained_bytes': sum(row['bytes'] for row in rows), 'by_archive': dict(counts_by_archive),
              'schema_invalid_samples': len(errors), 'samples_with_positive_mask': sum(row['positive_pixels_max'] > 0 for row in rows),
              'empty_mask_samples': sum(row['positive_pixels_max'] == 0 for row in rows),
              'official_ld_candidates': len(candidates), 'missing_official_candidates': len(missing),
              'exact_spectral_duplicate_groups': len(duplicates),
              'event_dates': dict(Counter(row['event_date'] for row in rows)),
              'crs_counts': dict(Counter(row['crs'] for row in rows)),
              'historical_40_10_210_split_recovered': False,
              'regional_completeness_claim': 'Only coverage of the pinned official LD candidate list is checked.',
              'seconds': time.monotonic() - started,
              'source_code_sha256': hash_file(__file__)}
    write_json(output / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
