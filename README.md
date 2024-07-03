# Tensorstore Extension for AnnData 

This extension provides support for reading and writing [AnnData](https://anndata.readthedocs.io/en/latest/) objects using the [Tensorstore](https://google.github.io/tensorstore/)

## Design Goals

- Large single-cell anndata object are often too large to fit in memory. Tensorstore provides a way to read and write data in chunks, allowing for out-of-core processing of large anndata objects.
- Tensorstore provides a way to read and write data in a variety of formats, including Zarr. This extension provides a way to read anndata objects with specific rows (cells) and columns (genes) from a Zarr store. **You will not need to load the entire anndata object into memory to access a subset of the data.**
- **Caveats**: This extension is still in development and may not support all features of AnnData objects.
- **Caveats**: This extension is not optimized for storage size. It is recommended to use the `anndata.write_zarr` method to write an anndata object to a Zarr store if storage size is a concern.



## Installation

```bash
pip install anndata-tensorstore
```

## Usage

### Writing an AnnData object to a Tensorstore

```python
import anndata
import anndata_tensorstore as ats

anndata = anndata.read_h5ad("path/to/large_anndata.h5ad")

# Create a tensorstore object from the anndata object
ats.save_anndata_to_tensorstore(anndata, "path/to/large_anndata.ats")
```


### Reading an AnnData object from a Tensorstore

```python
import os
import anndata
import anndata_tensorstore as ats

# Load the anndata object from the tensorstore, specifying the rows and columns to load
var = pd.read_parquet(os.path.join("path/to/large_anndata.ats", ats.ATS_FILE_NAME.var))

# Load the partial data from the tensorstore, specifying the rows and columns to load
adata = ats.load_anndata_from_tensorstore(
    "path/to/large_anndata.ats",
    obs=slice(0, 1000),                     # the specification of columns and rows can either be
    var=var.index.isin(["gene1", "gene2"])  # a slice object or a boolean array
)
```

## Development and Future Work

- [ ] reduce storage size
- [ ] support more AnnData features
- [ ] support more tensorstore features
