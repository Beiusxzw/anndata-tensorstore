import os
import tensorstore as ts
import pandas as pd
import numpy as np
import scipy
import pickle
from anndata import AnnData

from abc import ABC, ABCMeta
from enum import Enum, EnumMeta, unique
from functools import wraps
from typing import Any, Callable


class PrettyEnum(Enum):
    """Enum with a pretty :meth:`__str__` and :meth:`__repr__`."""

    @property
    def v(self) -> Any:
        """Alias for :attr`value`."""
        return self.value

    def __repr__(self) -> str:
        return f"{self.value!r}"

    def __str__(self) -> str:
        return f"{self.value!s}"
    
class ModeEnum(str, PrettyEnum, metaclass=EnumMeta):
    """Enum with a pretty :meth:`__str__` and :meth:`__repr__`."""

@unique
class ATS_FILE_NAME(ModeEnum):
    obs = 'obs.parquet'
    var = 'var.parquet'


def save_X(X, output_path, chunk_size=100):
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': output_path,
        },
        'metadata': {
            'dtype': X.dtype.str,
            'shape': X.shape
        },
    }, create=True, delete_existing=True).result()

    if scipy.sparse.issparse(X):
        for e in range(0,X.shape[0],chunk_size):
            write_future = dataset[e:min(X.shape[0], e+chunk_size), :].write(X[e:e+chunk_size].toarray())
            write_result = write_future.result()
    else:
        write_future = dataset.write(X)
        write_result = write_future.result()

def load_X(input_path, row_indices=None, col_indices=None):
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': input_path,
        }
    }, create=False).result()
    
    if row_indices is None and col_indices is None:
        return dataset.read().result()
    elif row_indices is not None and col_indices is None:
        return dataset[row_indices, :].read().result()
    elif row_indices is None and col_indices is not None:
        return dataset[:, col_indices].read().result()
    else:
        return dataset[row_indices, col_indices].read().result()

def save_np_array_to_tensorstore(Z, output_path):
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': output_path,
        },
        'metadata': {
            'dtype': Z.dtype.str,
            'shape': Z.shape
        },
    }, create=True).result()
    
    write_future = dataset.write(Z)
    write_result = write_future.result()


def load_np_array_from_tensorstore(input_path, row_indices=None):
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': input_path,
        }
    }, create=False).result()

    if row_indices is None:
        return dataset.read().result()
    else:
        return dataset[row_indices].read().result()


def check_is_parquet_serializable(obj: pd.DataFrame):
    for c in obj.columns:
        if obj[c].dtype == 'O':
            obj[c] = obj[c].astype(str)
        

def save_anndata_to_tensorstore(
    adata: AnnData,
    output_path: str
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if adata.X is not None:
       save_X(adata.X, os.path.join(output_path, 'X'))

    if adata.obs is not None:
        check_is_parquet_serializable(adata.obs)
        adata.obs.to_parquet(os.path.join(output_path, 'obs.parquet'))
    
    if adata.var is not None:
        check_is_parquet_serializable(adata.var)
        adata.var.to_parquet(os.path.join(output_path, 'var.parquet'))
    
    if adata.obsm is not None:
        for k, v in adata.obsm.items():
            save_np_array_to_tensorstore(v, os.path.join(output_path, f'obsm/{k}'))

    if adata.varm is not None:
        for k, v in adata.varm.items():
            save_np_array_to_tensorstore(v, os.path.join(output_path, f'varm/{k}'))
    
    if not os.path.exists(os.path.join(output_path, 'uns')):
        os.makedirs(os.path.join(output_path, 'uns'))

    if adata.uns is not None:
        for k, v in adata.uns.items():
            if isinstance(v, pd.DataFrame):
                check_is_parquet_serializable(v)
                v.to_parquet(os.path.join(output_path, f'uns/{k}.parquet'))
            else:
                with open(os.path.join(output_path, f'uns/{k}.pickle'), 'wb') as f:
                    pickle.dump(v, f)

    if adata.layers is not None:
        for k, v in adata.layers.items():
            save_X(v, os.path.join(output_path, f'layers/{k}'))

    if adata.raw is not None:
        save_anndata_to_tensorstore(adata.raw, os.path.join(output_path, 'raw'))

def load_anndata_from_tensorstore(input_path: str, row_indices=None, col_indices=None):
    _X = load_X(os.path.join(input_path, 'X'), row_indices, col_indices)
    

    if os.path.exists(os.path.join(input_path, 'obs.parquet')):
        obs = pd.read_parquet(os.path.join(input_path, 'obs.parquet'))
        # if row_indices is a slice
        if isinstance(row_indices, slice):
            _obs = obs.iloc[row_indices]
        # if row_indices is a array
        elif isinstance(row_indices, np.ndarray):
            # if row_indices is a boolean array
            if row_indices.dtype == np.bool:
                _obs = obs.loc[row_indices]
            else:
                _obs = obs.iloc[row_indices]
        elif row_indices is None:
            _obs = obs
    
    if os.path.exists(os.path.join(input_path, 'var.parquet')):
        var = pd.read_parquet(os.path.join(input_path, 'var.parquet'))
        # if col_indices is a slice
        if isinstance(col_indices, slice):
            _var = var.iloc[col_indices]
        # if col_indices is a array
        elif isinstance(col_indices, np.ndarray):
            # if col_indices is a boolean array
            if col_indices.dtype == bool:
                _var = var.loc[col_indices]
            else:
                _var = var.iloc[col_indices]
        elif col_indices is None:
            _var = var
    _obsm = None
    if os.path.exists(os.path.join(input_path, 'obsm')):
        _obsm = {}
        for f in os.listdir(os.path.join(input_path, 'obsm')):
            _obsm[f] = load_np_array_from_tensorstore(os.path.join(input_path, 'obsm', f), row_indices)
    
    _varm = None
    if os.path.exists(os.path.join(input_path, 'varm')):
        _varm = {}
        for f in os.listdir(os.path.join(input_path, 'varm')):
            _varm[f] = load_np_array_from_tensorstore(os.path.join(input_path, 'varm', f), col_indices)
    
    _uns = None
    if os.path.exists(os.path.join(input_path, 'uns')):
        _uns = {}
        for fname in os.listdir(os.path.join(input_path, 'uns')):
            if fname.endswith('.parquet'):
                _uns[fname[:-8]] = pd.read_parquet(os.path.join(input_path, 'uns', fname))
            elif fname.endswith('.pickle'):
                with open(os.path.join(input_path, 'uns', fname), 'rb') as f:
                    _uns[fname] = pickle.load(f)
    
    _layers = None
    if os.path.exists(os.path.join(input_path, 'layers')):
        _layers = {}
        for f in os.listdir(os.path.join(input_path, 'layers')):
            _layers[f] = load_X(os.path.join(input_path, 'layers', f), row_indices, col_indices)
    
    adata = AnnData(
        X=_X,
        obs=_obs,
        var=_var,
        obsm=_obsm,
        varm=_varm,
        uns=_uns,
        layers=_layers
    )
    if os.path.exists(os.path.join(input_path, 'raw')):
        adata.raw = load_anndata_from_tensorstore(os.path.join(input_path, 'raw'), row_indices, col_indices)
    
    return adata