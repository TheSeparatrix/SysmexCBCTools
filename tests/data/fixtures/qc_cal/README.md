# Test fixture: QC_CAL_sample

Quality-control material, not patient blood: every sample number is `QC-…` and the
patient columns are empty. `tests/data/test_fixture_safety.py` fails if anything else
lands here.

`XN_SAMPLE.csv` (2 measurements, every column) is copied byte for byte from
bloodcounts-sysmex-format's `tests/fixtures/export/`, the slice raw_data_preparation
vendors too. To refresh it, copy it from there again. Never put a real export here.
