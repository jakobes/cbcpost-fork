#!/usr/bin/env py.test

import pytest

from cbcpost import ParamDict

def test_init_from_dict():
    d = { 'a': 1, 'b': 3.14 }
    pd = ParamDict(d)
    assert len(pd) == 2
    assert pd.a == d['a']
    assert pd.b == d['b']
    assert pd['a'] == d['a']
    assert pd['b'] == d['b']

def test_init_by_kwargs():
    pd = ParamDict(foo='hei', bar='argh')
    assert pd.foo == 'hei'
    assert pd.bar == 'argh'
    assert pd["foo"] == 'hei'
    assert pd["bar"] == 'argh'
    assert len(pd) == 2

def test_init_by_sequence():
    keys = ('a', 'b', 'c')
    values = (1, 2, 3)
    items = tuple(zip(keys, values))
    pd = ParamDict(items)
    assert pd.a == 1
    assert pd.b == 2
    assert pd.c == 3

def test_add_params_after_init_succeeds_with_dict_notation():
    pd = ParamDict()
    pd["a"] = 1
    pd["b"] = 2
    assert pd.a == 1
    assert pd.b == 2
    assert len(pd) == 2

def test_add_params_after_init_raises_with_attribute_notation():
    pd = ParamDict()
    with pytest.raises(RuntimeError):
        pd.a = 1

def test_shallow_iteration():
    keys = ('a', 'b', 'c')
    values = (1, 2, 3)
    items = tuple(zip(keys, values))
    pd = ParamDict(items)
    assert tuple(sorted(pd)) == keys
    assert tuple(sorted(pd.iterkeys())) == keys
    assert tuple(sorted(pd.keys())) == keys
    assert tuple(sorted(pd.iteritems())) == items
    assert tuple(sorted(pd.itervalues())) == values

def create_multilevel_pd():
    pda1 = ParamDict(a=1)
    pdb1 = ParamDict(b=2)
    pdc1 = ParamDict(pa=pda1, pb=pdb1)
    pda2 = ParamDict(a=3)
    pdb2 = ParamDict(b=4)
    pdc2 = ParamDict(pa=pda2, pb=pdb2)
    pdd = ParamDict(pc1=pdc1, pc2=pdc2)
    return pdd

def test_multilevel_access():
    pdd = create_multilevel_pd()
    assert pdd.pc1.pa.a == 1
    assert pdd.pc1.pb.b == 2
    assert pdd.pc2.pa.a == 3
    assert pdd.pc2.pb.b == 4

def test_iterdeep_shallow_data():
    pd = ParamDict()
    deep = tuple(pd.iterdeep())
    assert deep == ()

    pd = ParamDict(a=3, b=4)
    deep = tuple(pd.iterdeep())
    assert deep == (('a',3), ('b',4))

def test_iterdeep_multilevel_data():
    pdd = create_multilevel_pd()
    deep = tuple(sorted(pdd.iterdeep()))
    items = ( ('pc1.pa.a', 1),
              ('pc1.pb.b', 2),
              ('pc2.pa.a', 3),
              ('pc2.pb.b', 4), )
    assert deep == items

def test_shallow_copy():
    pd1 = ParamDict(a=3, b=4)
    pd2 = pd1.copy_recursive()
    pd1.a = 1
    pd2.a = 2
    pd2.b = 2
    pd1.b = 1
    assert pd1.a == 1
    assert pd1.b == 1
    assert pd2.a == 2
    assert pd2.b == 2

def test_recursive_copy():
    pdcc1 = ParamDict(cca=30)
    pdcc2 = ParamDict(ccb=40)
    pdc1 = ParamDict(a=3, b=4, cc1=pdcc1, cc2=pdcc2)
    pdc2 = ParamDict(c=5, d=6)
    pd1 = ParamDict(c1=pdc1, c2=pdc2)
    pd2 = pd1.copy_recursive()

    assert pd1.c2.d == 6
    assert pd2.c2.d == 6
    pd1.c2.d = 7
    assert pd1.c2.d == 7
    assert pd2.c2.d == 6
    pd2.c2.d = 8
    assert pd1.c2.d == 7
    assert pd2.c2.d == 8

    assert pd1.c1.cc2.ccb == 40
    assert pd2.c1.cc2.ccb == 40
    pd2.c1.cc2.ccb = 50
    assert pd1.c1.cc2.ccb == 40
    assert pd2.c1.cc2.ccb == 50
    pd1.c1.cc2.ccb = 60
    assert pd1.c1.cc2.ccb == 60
    assert pd2.c1.cc2.ccb == 50

def test_shallow_replace():
    pd1 = ParamDict(a=3, b=4)
    pd2 = ParamDict(b=14)
    pd3 = ParamDict(c=15)
    pd1orig = pd1.copy()
    pd1.replace_shallow(pd2)
    assert all(k in pd1 for k in pd1orig)
    assert all(pd1[k] == pd2[k] for k in pd2)
    assert all(pd1[k] == pd1orig[k] for k in pd1orig if not k in pd2)
    with pytest.raises(RuntimeError):
        pd1.replace_shallow(pd3)

def test_recursive_replace():
    # Build multilevel test data
    pdcc1 = ParamDict(cca=30)
    pdcc2 = ParamDict(ccb=40)
    pdc1 = ParamDict(a=3, b=4, cc1=pdcc1, cc2=pdcc2)
    pdc2 = ParamDict(c=5, d=6)
    pdorig = ParamDict(c1=pdc1, c2=pdc2, v=7)

    # Build alternative multilevel test data
    apdcc1 = ParamDict(cca=31)
    apdcc2 = ParamDict(ccb=41)
    apdc1 = ParamDict(a=5, b=8, cc1=apdcc1, cc2=apdcc2)
    apdc2 = ParamDict(c=7, d=9)
    apdorig = ParamDict(c1=apdc1, c2=apdc2, v=2)

    assert pdorig != apdorig

    # Supply a single item
    pd = pdorig.copy_recursive()
    assert pd.v == 7
    pd.replace_recursive(v=9)
    assert pd.v == 9
    pd.replace_recursive({'v':10})
    assert pd.v == 10
    pd.replace_recursive(ParamDict(v=11))
    assert pd.v == 11

    # Supply multiple items for child
    pd = pdorig.copy_recursive()
    assert pd.c1.a == 3
    assert pd.c1.b == 4
    if 0:
        pd.replace_recursive(c1={'a':11, 'b':22})
        assert pd.c1.a == 11
        assert pd.c1.b == 22
    pd.replace_recursive(c1=ParamDict(a=13, b=25))
    assert pd.c1.a == 13
    assert pd.c1.b == 25

    # Supply a full multilevel paramdict
    pd = pdorig.copy_recursive()
    assert pd == pdorig
    assert pd != apdorig
    pd.replace_recursive(apdorig)
    assert pd != pdorig
    assert pd == apdorig

    # Raises for single missing item
    pd = pdorig.copy_recursive()
    with pytest.raises(RuntimeError):
        pd.replace_recursive(v2=13)

    # Build alternative multilevel test data
    rpdcc1 = ParamDict(cca2=32)
    rpdc1 = ParamDict(cc1=rpdcc1)
    rpdorig = ParamDict(c1=rpdc1)

    # Raises for item deep in recursive structure
    pd = pdorig.copy_recursive()
    with pytest.raises(RuntimeError):
        pd.replace_recursive(rpdorig)

def test_shallow_update():
    pd1 = ParamDict(a=3, b=4)
    pd2 = ParamDict(b=14, c=15)
    pd1orig = pd1.copy()
    pd1.update_shallow(pd2)
    assert all(k in pd1 for k in pd1orig)
    assert all(k in pd1 for k in pd2)
    assert all(pd1[k] == pd2[k] for k in pd2)
    assert all(pd1[k] == pd1orig[k] for k in pd1orig if not k in pd2)

def test_recursive_update():
    # Build multilevel test data
    pdcc1 = ParamDict(cca=30)
    pdcc2 = ParamDict(ccb=40)
    pdc1 = ParamDict(a=3, b=4, cc1=pdcc1, cc2=pdcc2)
    pdc2 = ParamDict(c=5, d=6)
    pdorig = ParamDict(c1=pdc1, c2=pdc2, v=7)

    # Build alternative multilevel test data
    apdcc1 = ParamDict(cca=31)
    apdcc2 = ParamDict(ccb=41)
    apdc1 = ParamDict(a=5, b=8, cc1=apdcc1, cc2=apdcc2)
    apdc2 = ParamDict(c=7, d=9)
    apdorig = ParamDict(c1=apdc1, c2=apdc2, v=2)

    assert pdorig != apdorig

    # Supply a single item
    pd = pdorig.copy_recursive()
    assert pd.v == 7
    pd.update_recursive(v=9)
    assert pd.v == 9
    pd.update_recursive({'v':10})
    assert pd.v == 10
    pd.update_recursive(ParamDict(v=11))
    assert pd.v == 11

    # Supply multiple items for child
    pd = pdorig.copy_recursive()
    assert pd.c1.a == 3
    assert pd.c1.b == 4
    if 0:
        pd.update_recursive(c1={'a':11, 'b':22})
        assert pd.c1.a == 11
        assert pd.c1.b == 22
    pd.update_recursive(c1=ParamDict(a=13, b=25))
    assert pd.c1.a == 13
    assert pd.c1.b == 25

    # Supply a full multilevel paramdict
    pd = pdorig.copy_recursive()
    assert pd == pdorig
    assert pd != apdorig
    pd.update_recursive(apdorig)
    assert pd != pdorig
    assert pd == apdorig

def test_pickle_protocol():
    pass # TODO

def test_repr_rendering():
    pass # TODO

def test_str_rendering():
    pdid = ParamDict(a=1, b=2)
    pdin = ParamDict(a=1, b=3)
    pdout = ParamDict(a=1, b=4)
    record = ParamDict(identity=pdid,
                       input=pdin,
                       output=pdout)
    s = str(record)
    assert record["identity"] == pdid

def test_arg_rendering_parsing():
    pdid = ParamDict(a=1, b=2)
    pdin = ParamDict(a=1, b=3)
    pdout = ParamDict(a=1, b=4)
    record = ParamDict(a="foo", b="bar", c=(1,2,"foo"),
                       identity=pdid,
                       input=pdin,
                       output=pdout)

    pd_default = ParamDict(a=None,b=None)
    pd = ParamDict(identity=pd_default,
                   input=pd_default,
                   output=pd_default,
                   c = None,
                   **pd_default)

    assert pd != record
    pd.parse_args(record.render_args())
    assert pd == record
