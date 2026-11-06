"""Native WN18RR TSV adapter, preserving public synset keys and aligned images."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import re
from pathlib import Path

from mmkgc.data.mkgw import MKGWDataStore

WN18RR_SPLIT_FILES = {
    'train': ('train.tsv',),
    'train-valid': ('train.tsv', 'dev.tsv'),
    'all': ('train.tsv', 'dev.tsv', 'test.tsv'),
}
WN18RR_REQUIRED_FILES = ('entities.txt', 'relations.txt', 'entity2text.txt', 'train.tsv')


def _rows(path: Path, columns: int):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for number, row in enumerate(csv.reader(handle, delimiter='\t', quoting=csv.QUOTE_NONE), 1):
            if not row or not any(part.strip() for part in row):
                continue
            if len(row) != columns or any(not part.strip() for part in row):
                raise ValueError(f'{path.name}:{number}: expected {columns} nonempty tab-separated fields')
            yield number, [part.strip() for part in row]


def _mapping(path: Path):
    keys = []
    seen = set()
    for number, (key,) in _rows(path, 1):
        if key in seen:
            raise ValueError(f'{path.name}:{number}: duplicate key {key}')
        keys.append(key)
        seen.add(key)
    if not keys:
        raise ValueError(f'{path.name} is empty')
    return dict(enumerate(keys)), {key: index for index, key in enumerate(keys)}


class WN18RRDataStore(MKGWDataStore):
    """Reuse the canonical record/masking/traversal contract with native TSV input."""

    def __init__(self, dataset_dir: str | Path, graph_split: str = 'train'):
        if graph_split not in WN18RR_SPLIT_FILES:
            raise ValueError('graph_split must be one of: train, train-valid, all')
        self.dataset_dir = Path(dataset_dir)
        self.graph_split = graph_split
        self.graph_filenames = WN18RR_SPLIT_FILES[graph_split]
        self.entity_id_to_uri, self.entity_uri_to_id = _mapping(self.dataset_dir / 'entities.txt')
        self.relation_id_to_uri, self.relation_uri_to_id = _mapping(self.dataset_dir / 'relations.txt')
        if any(not re.fullmatch(r'\d{8}', key) for key in self.entity_uri_to_id):
            raise ValueError('WN18RR entity keys must be eight-digit WordNet offsets')
        self.entity_info = {}
        definitions = self.dataset_dir / 'wordnet-mlj12-definitions.txt'
        if definitions.exists():
            for number, (key, name, _definition) in _rows(definitions, 3):
                if key not in self.entity_uri_to_id:
                    raise ValueError(f'{definitions.name}:{number}: unknown entity {key}')
                match = re.fullmatch(r'__(.+)_(?:NN|VB|JJ|RB)_\d+', name)
                if not match:
                    raise ValueError(f'{definitions.name}:{number}: invalid synset name {name}')
                if key in self.entity_info:
                    raise ValueError(f'{definitions.name}:{number}: duplicate entity {key}')
                # Only the independently supplied name is public; never import its definition.
                self.entity_info[key] = {'label': match[1].replace('_', ' ')}
        self.entity_texts = self._text_mapping('entity2text.txt', self.entity_uri_to_id)
        self.relation_info = {}
        if (self.dataset_dir / 'relation2text.txt').exists():
            self.relation_info = self._text_mapping('relation2text.txt', self.relation_uri_to_id)
        index_path = self.dataset_dir / 'image_alignment' / 'entity2images.json'
        self._aligned_images = json.loads(index_path.read_text()) if index_path.exists() else None
        if self._aligned_images is not None and not isinstance(self._aligned_images, dict):
            raise ValueError('entity2images.json must be an object')
        self._image_indexes = []
        self.triples = []
        for filename in self.graph_filenames:
            self.triples.extend(self._load_triples(filename))
        self._adjacency = self._build_directed_adjacency()

    def _text_mapping(self, name, keys):
        result = {}
        for number, (key, text) in _rows(self.dataset_dir / name, 2):
            if key not in keys or key in result:
                raise ValueError(f'{name}:{number}: unknown or duplicate key {key}')
            result[key] = text
        return result

    def _load_triples(self, filename):
        triples = []
        for number, (head, relation, tail) in _rows(self.dataset_dir / filename, 3):
            if head not in self.entity_uri_to_id or tail not in self.entity_uri_to_id:
                raise ValueError(f'{filename}:{number}: unknown triple entity')
            if relation not in self.relation_uri_to_id:
                raise ValueError(f'{filename}:{number}: unknown triple relation {relation}')
            triples.append((self.entity_uri_to_id[head], self.relation_uri_to_id[relation], self.entity_uri_to_id[tail]))
        return triples

    def image_paths(self, entity_id):
        if self._aligned_images is None:
            return super().image_paths(entity_id)
        key = self.entity_key(entity_id)
        # An audited empty/missing alignment must not fall back to excluded raw directories.
        values = self._aligned_images.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f'Invalid image alignment for {key}')
        root = self.dataset_dir.resolve()
        paths = []
        for value in values:
            path = (root / value).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Image alignment escapes dataset_dir for {key}')
            if not path.is_file():
                raise FileNotFoundError(f'Aligned image does not exist: {path}')
            if str(path) not in paths:
                paths.append(str(path))
        return paths

    def build_record(self, entity_id, missing_mode):
        record = super().build_record(entity_id, missing_mode)
        if missing_mode == 'text':
            def mask_self(nodes):
                for node in nodes:
                    if node.get('entity_key') == record.entity_key:
                        node['description'] = node['neighbor_description'] = ''
                    mask_self(node.get('neighbors', []))
            mask_self(record.neighbors)
            record = replace(record, description='')
        return replace(record, relation_semantics={
            self.relation_label(self.relation_uri_to_id[key]): meaning
            for key, meaning in WN18RR_RELATION_MEANINGS.items() if key in self.relation_uri_to_id})


# Direction is always original subject -> original object, independent of the
# target's in/out flag. WordNet pointer semantics: https://wordnet.princeton.edu/documentation/wninput5wn
WN18RR_RELATION_MEANINGS = {
    '_hypernym': 'The subject has the object as its broader category: subject is a kind of object.',
    '_instance_hypernym': 'The subject is an individual instance of the object category.',
    '_has_part': 'The subject whole has the object as a part.',
    '_member_meronym': 'The subject group or whole has the object as a member.',
    '_synset_domain_topic_of': 'The subject sense belongs to the object topic domain.',
    '_member_of_domain_usage': 'The subject usage/register domain contains the object term.',
    '_member_of_domain_region': 'The subject regional domain contains the object term.',
    '_derivationally_related_form': 'Subject and object word forms have a derivational lexical link.',
    '_similar_to': 'Subject and object senses are semantically similar; no physical resemblance is asserted.',
    '_also_see': 'A lexical cross-reference links the subject sense to the object sense.',
    '_verb_group': 'The subject and object verb senses belong to a semantic verb group.',
}
