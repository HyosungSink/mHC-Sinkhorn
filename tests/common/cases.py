"""Load disjoint device suites; full excludes performance Mock cases."""
import json
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1] / 'st/cases'
CATEGORIES = ('coverage', 'correctness', 'mock')


def load_cases(suite='full', names=None, tags=None):
    if suite not in (*CATEGORIES, 'full', 'custom'):
        raise ValueError(f'Unknown suite: {suite}')
    categories = CATEGORIES if suite == 'custom' else (
        ('coverage', 'correctness') if suite == 'full' else (suite,))
    rows = []
    seen = set()
    for category in categories:
        payload = json.loads((CASE_ROOT / f'{category}.json').read_text())
        if payload['category'] != category:
            raise ValueError(f'Incorrect category: {category}')
        for case in payload['cases']:
            if case['name'] in seen:
                raise ValueError(f'Duplicate case: {case["name"]}')
            seen.add(case['name'])
            rows.append(case)
    names = set(names or [])
    if suite == 'custom' and not names:
        raise ValueError('custom requires --case')
    if names - seen:
        raise ValueError(f'Cases not in selected suite: {sorted(names - seen)}')
    tags = set(tags or [])
    selected = [case for case in rows if (not names or case['name'] in names)
                and tags.issubset(case.get('tags', []))]
    if not selected:
        raise ValueError('No cases match the selected suite, names and tags')
    return selected
