import numpy as np
import pytest
from edgeai.worker import topk_counts


def test_topk():
    scores=np.array([[.1,.9,.0],[.7,.2,.1],[.0,.0,1.0]])
    assert topk_counts(scores,np.array([1,2,2]),k=2)==(2,2)

def test_topk_wrong_dimensions():
    with pytest.raises(ValueError):topk_counts(np.array([.1,.9]),np.array([1]))

def test_topk_invalid_label():
    with pytest.raises(ValueError):topk_counts(np.zeros((2,3)),np.array([0,9]))

def test_topk_top5_when_few_classes():
    s=np.array([[.5,.2,.1],[.1,.8,.3]])
    assert topk_counts(s,np.array([2,1]),k=5)==(1,2)
