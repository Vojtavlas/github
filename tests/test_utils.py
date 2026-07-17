import re
from unittest.mock import MagicMock, patch

import pytest
import torch

from reasonflow.utils import load_model_and_tokenizer, squeeze_hidden


def _make_tok(pad_token=None, eos_token="</s>", eos_token_id=2):
    tok = MagicMock()
    tok.pad_token = pad_token
    tok.eos_token = eos_token
    tok.eos_token_id = eos_token_id
    return tok


def test_squeeze_hidden_1d():
    h = torch.tensor([1.0, 2.0, 3.0])
    out = squeeze_hidden(h)
    assert torch.equal(out, h)
    assert out.shape == (3,)


def test_squeeze_hidden_2d():
    h = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = squeeze_hidden(h)
    assert torch.equal(out, h[-1, :])
    assert out.shape == (2,)


def test_squeeze_hidden_3d_batch_one():
    h = torch.arange(12).reshape(1, 4, 3).float()
    out = squeeze_hidden(h)
    assert torch.equal(out, h[0, -1, :])
    assert out.shape == (3,)


def test_squeeze_hidden_3d_batch_one_tuple():
    h = torch.arange(12).reshape(1, 4, 3).float()
    out = squeeze_hidden((h, None))
    assert torch.equal(out, h[0, -1, :])


def test_squeeze_hidden_3d_batch_gt_one_raises():
    h = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match=re.escape("batch size 1")):
        squeeze_hidden(h)


@patch("reasonflow.utils.AutoModelForCausalLM")
@patch("reasonflow.utils.AutoTokenizer")
def test_load_model_and_tokenizer_cpu(mock_tokenizer_cls, mock_model_cls):
    tok = _make_tok(pad_token=None, eos_token="</s>", eos_token_id=2)
    mock_tokenizer_cls.from_pretrained.return_value = tok
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model_cls.from_pretrained.return_value = mock_model

    model, returned_tok = load_model_and_tokenizer("dummy-model", device="cpu")

    assert returned_tok is tok
    assert tok.pad_token == "</s>"
    assert tok.pad_token_id == 2
    assert tok.padding_side == "left"
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "dummy-model", trust_remote_code=True
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "dummy-model",
        dtype=torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    mock_model.to.assert_called_once_with("cpu")
    mock_model.eval.assert_called_once()
    assert model is mock_model


@patch("reasonflow.utils.AutoModelForCausalLM")
@patch("reasonflow.utils.AutoTokenizer")
def test_load_model_and_tokenizer_cuda1(mock_tokenizer_cls, mock_model_cls):
    tok = _make_tok(pad_token=None, eos_token="</s>", eos_token_id=2)
    mock_tokenizer_cls.from_pretrained.return_value = tok
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model_cls.from_pretrained.return_value = mock_model

    model, returned_tok = load_model_and_tokenizer("dummy-model", device="cuda:1")

    assert returned_tok is tok
    mock_model_cls.from_pretrained.assert_called_once_with(
        "dummy-model",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="cuda:1",
    )
    mock_model.to.assert_not_called()
    mock_model.eval.assert_called_once()
    assert model is mock_model


@patch("reasonflow.utils.AutoModelForCausalLM")
@patch("reasonflow.utils.AutoTokenizer")
def test_load_model_and_tokenizer_cuda_auto(mock_tokenizer_cls, mock_model_cls):
    tok = _make_tok(pad_token=None, eos_token="</s>", eos_token_id=2)
    mock_tokenizer_cls.from_pretrained.return_value = tok
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model_cls.from_pretrained.return_value = mock_model

    load_model_and_tokenizer("dummy-model", device="cuda")

    call_kwargs = mock_model_cls.from_pretrained.call_args.kwargs
    assert call_kwargs["device_map"] == "auto"
    assert call_kwargs["dtype"] is torch.bfloat16


@patch("reasonflow.utils.AutoModelForCausalLM")
@patch("reasonflow.utils.AutoTokenizer")
def test_load_model_and_tokenizer_missing_pad_and_eos_raises(
    mock_tokenizer_cls, mock_model_cls
):
    tok = _make_tok(pad_token=None, eos_token=None, eos_token_id=None)
    mock_tokenizer_cls.from_pretrained.return_value = tok

    with pytest.raises(ValueError, match="no pad_token or eos_token"):
        load_model_and_tokenizer("dummy-model", device="cpu")
