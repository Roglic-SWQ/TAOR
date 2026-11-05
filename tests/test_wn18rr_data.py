import json
from pathlib import Path

import pytest

from mmkgc.data import DatasetStore, WN18RRDataStore, build_dataset_store
from mmkgc.data.contracts import existing_image_paths
from mmkgc.run import build_parser, validate_dataset_dir
from mmkgc.stages.sources import public_sources


def fixture(tmp_path):
    files = {
        'entities.txt': '00000001\n00000002\n00000003\n',
        'relations.txt': '_hypernym\n_has_part\n',
        'entity2text.txt': '00000001\tHIDDEN description of target\n00000002\tNeighbor text\n',
        'relation2text.txt': '_hypernym\thypernym\n_has_part\thas part\n',
        'wordnet-mlj12-definitions.txt': '00000001\t__target_lemma_NN_1\tPRIVATE definition\n00000002\t__neighbor_NN_1\tNeighbor definition\n',
        'train.tsv': '00000002\t_hypernym\t00000001\n00000001\t_has_part\t00000003\n',
        'dev.tsv': '00000001\t_hypernym\t00000003\n',
        'test.tsv': '00000003\t_has_part\t00000002\n',
    }
    for name,text in files.items(): (tmp_path/name).write_text(text)
    (tmp_path/'images/00000001').mkdir(parents=True)
    for name in ['b.JPEG','a.JPEG']:
        (tmp_path/'images/00000001'/name).write_bytes(b'original')
    (tmp_path/'images/00000002').mkdir(parents=True)
    (tmp_path/'images/00000002/excluded.JPEG').write_bytes(b'excluded')
    (tmp_path/'image_alignment').mkdir()
    (tmp_path/'image_alignment/entity2images.json').write_text(json.dumps({
        '00000001':['images/00000001/b.JPEG','images/00000001/a.JPEG'],
        '00000002':[]}))
    return tmp_path


def test_native_factory_public_keys_direction_and_train_only(tmp_path):
    store=build_dataset_store('wn18rr',fixture(tmp_path))
    assert isinstance(store, DatasetStore)
    record=store.build_record(0,'image')
    assert record.entity_key=='00000001' and record.label=='target lemma'
    assert len(store.triples)==2
    assert [(n['entity_key'],n['relation'],n['direction']) for n in record.neighbors] == [
        ('00000002','hypernym','in'), ('00000003','has part','out')]
    assert [r.entity_key for r in store.select_records_by_keys('text',['00000003','00000001'])] == ['00000003','00000001']
    with pytest.raises(ValueError,match='unknown entity_key'):
        store.select_records_by_keys('image',['1'])


@pytest.mark.parametrize('split,count',[('train',2),('train-valid',3),('all',4)])
def test_split_selection_is_explicit(tmp_path,split,count):
    store=WN18RRDataStore(fixture(tmp_path),graph_split=split)
    assert len(store.triples)==count


def test_masking_uses_names_but_never_definition_or_target_images(tmp_path):
    store=WN18RRDataStore(fixture(tmp_path))
    image=store.build_record(0,'image')
    assert existing_image_paths(image)==[]
    assert image.existing_modalities['image'] is None
    assert image.modality_completion_policy['image']['status']=='masked_for_eval'
    assert 'HIDDEN' in image.existing_modalities['text']
    text=store.build_record(0,'text')
    sources,catalog=public_sources(text,'text')
    assert 'HIDDEN' not in json.dumps((sources,catalog))
    assert 'PRIVATE' not in json.dumps((sources,catalog))
    assert text.description=='' and text.label=='target lemma'
    assert len(existing_image_paths(text))==2
    assert text.ground_truth_modalities['text']=='HIDDEN description of target'
    assert store.build_record(2,'text').modality_completion_policy['text']['status']=='real_missing'


def test_authoritative_missing_alignment_does_not_rebind_excluded_images(tmp_path):
    store=WN18RRDataStore(fixture(tmp_path))
    assert store.image_paths(1)==[]
    assert store.image_paths(2)==[]
    assert store.build_record(1,'image').modality_completion_policy['image']['status']=='real_missing'


@pytest.mark.parametrize('bad',[['../outside.jpg'],['/etc/passwd'],['images/missing.JPEG'],'not-a-list',[123]])
def test_bad_alignment_fails_explicitly(tmp_path,bad):
    root=fixture(tmp_path)
    (root/'image_alignment/entity2images.json').write_text(json.dumps({'00000001':bad}))
    with pytest.raises((ValueError,FileNotFoundError)):
        WN18RRDataStore(root).build_record(0,'text')


@pytest.mark.parametrize('filename,bad',[
    ('train.tsv','00000001\t_unknown\t00000002\n'),
    ('train.tsv','00000001\t_hypernym\t99999999\n'),
    ('train.tsv','00000001 _hypernym 00000002\n'),
    ('entities.txt','00000001\n00000001\n'),
    ('entities.txt','1\n00000002\n00000003\n'),
    ('entity2text.txt','99999999\tunknown entity text\n'),
])
def test_malformed_graph_and_mapping_are_not_silently_dropped(tmp_path,filename,bad):
    root=fixture(tmp_path)
    (root/filename).write_text(bad)
    with pytest.raises(ValueError): WN18RRDataStore(root)


def test_cli_and_doctor_share_native_dataset_validation(tmp_path):
    root=fixture(tmp_path)
    args=build_parser().parse_args(['--dataset-type','wn18rr','--dataset-dir',str(root)])
    assert validate_dataset_dir(root,args.dataset_type)==root
    with pytest.raises(ValueError,match='entity2id.txt'): validate_dataset_dir(root)
    (root/'train.tsv').write_text('')
    with pytest.raises(ValueError,match='train.tsv'):validate_dataset_dir(root,'wn18rr')


def test_doctor_accepts_wn18rr_without_mkgw_mapping_files(tmp_path):
    from mmkgc.doctor import build_parser as doctor_parser, _check_dataset
    root=fixture(tmp_path)
    args=doctor_parser().parse_args(['--dataset-type','wn18rr','--dataset-dir',str(root)])
    checks=[]
    assert _check_dataset(args,checks)==root
    assert checks==[{'name':'dataset_dir','status':'ok','message':'wn18rr dataset is readable'}]


def test_text_masking_does_not_leak_through_self_loop(tmp_path):
    root=fixture(tmp_path)
    with (root/'train.tsv').open('a') as f:
        f.write('00000001\t_hypernym\t00000001\n')
    record=WN18RRDataStore(root).build_record(0,'text')
    own=[n for n in record.neighbors if n['entity_key']==record.entity_key]
    assert own and all(n['description']==n['neighbor_description']=='' for n in own)
