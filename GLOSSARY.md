# SysmexCBCTools

Tools for cleaning, aligning and correcting Sysmex XN-series CBC data. Sysmex vocabulary (HAAS output, Sysmex channel, scattergram, measurement, dataset, read API) follows the [bloodcounts-sysmex-format glossary](https://github.com/bloodcounts/bloodcounts-sysmex-format/blob/main/CONTEXT.md).

## Input routes

**Sysmex dataset**:
A dataset in the BloodCounts-internal bloodcounts-sysmex-format: tabular results and scattergrams for many measurements, read through its read API.
_Avoid_: DuckDB dataset, database, dataset (alone: `dataset_name` already means an output label here)

**HAAS route**:
Reading HAAS output files (`XN_SAMPLE.csv`, `OutputData.csv`, scattergram CSVs) straight from disk, with no dataset involved. The default route.
_Avoid_: CSV mode, file route

**Direct route**:
Reading a bloodcounts-sysmex-format dataset through its read API.
_Avoid_: DuckDB route, DuckDB source, database mode

**Export route**:
Running `bc-sysmex-format export` on a dataset, then reading the regenerated HAAS output by the HAAS route.

## Provenance

**Processing record**:
The account of one cleaning run that lets a researcher see how each measurement was processed: the settings and inputs, every row dropped and why, every column removed and why, and the outcome.
_Avoid_: audit log (in bloodcounts-sysmex-format that means proof of what was done to patient data at a site)
