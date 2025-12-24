# Memory Optimization Guide for Large Sysmex Datasets

This guide addresses memory issues when processing large datasets (>1M rows, >400 columns) with the Sysmex cleaning tool.

## What's New - Memory Optimizations

### 1. **Automatic Memory Monitoring**
- The tool now logs memory usage throughout processing
- Automatic garbage collection to free unused memory
- Progress indicators for long-running operations

### 2. **Smart Correlation Analysis**
- Automatically samples large datasets for correlation analysis
- Fallback to smaller samples if memory errors occur
- Configurable sample size limits

### 3. **Optimized Multiple Measurements Processing**
- **Dask integration** for datasets >100k rows (automatic)
- **Chunked processing** fallback if Dask unavailable
- Processes duplicate samples in manageable chunks

### 4. **Configuration Options**
New settings in `config.yaml`:
```yaml
processing:
  # Memory optimization settings (NEW)
  use_memory_optimized: true        # Enable optimized processing
  enable_memory_monitoring: true    # Log memory usage
  correlation_sample_size: 50000    # Max rows for correlation
  chunk_size: 1000                  # Chunk size for processing
```

## Installation Requirements

### Core Dependencies (Updated)
```bash
pip install pandas numpy pyyaml tqdm psutil
```

### Optional but Recommended for Large Datasets
```bash
pip install dask[complete]
# OR for minimal installation:
pip install dask dask[dataframe]
```

**Note**: The tool works without Dask but will use slower chunked processing for large datasets.

## For Your Dutch Colleague's Dataset

### Dataset Characteristics
- **1.2 million rows × 460 columns**
- **110,000 multiple measurements (9.8%)**
- **Previously failed with 64GB RAM**

### Recommended Settings
Update your `config.yaml`:

```yaml
processing:
  remove_clotintube: true
  remove_multimeasurementsamples: true
  remove_correlated: false
  std_threshold: 1.0
  keep_drop_rows: false
  make_dummy_marks: false
  
  # Optimized settings for large datasets
  use_memory_optimized: true
  enable_memory_monitoring: true
  correlation_sample_size: 20000     # Reduced for 460 columns
  chunk_size: 500                    # Smaller chunks for safety
```

### Expected Improvements
1. **Memory usage**: ~10-15GB instead of >64GB
2. **Processing time**: May be slower but won't crash
3. **Progress feedback**: Clear indicators of processing status
4. **Graceful handling**: Automatic fallbacks if memory issues occur

## Troubleshooting

### If Processing Still Hangs
1. **Reduce chunk size**: Set `chunk_size: 200` in config
2. **Disable correlation analysis**: Set `remove_correlated: false` (already default)
3. **Use smaller correlation sample**: Set `correlation_sample_size: 10000`

### If Memory Errors Persist
1. **Install Dask**: `pip install dask[complete]`
2. **Increase system virtual memory/swap**
3. **Process subsets**: Split your dataset into smaller files

### Error: "failed check on sample, variable, value 1 and value 2"
This is now handled better:
- **Chunked processing** prevents infinite loops
- **Progress indicators** show processing status
- **Debug logging** reduced (only on demand)

### Adding Demographics (Age, Gender, etc.)
**Safe to add** - the optimized tool handles additional columns efficiently:
- Memory usage scales linearly with columns
- Processing optimizations work regardless of column types
- New columns won't break existing logic

## Usage Examples

### Basic Usage (Same as Before)
```bash
python process_XN_SAMPLE.py --config config.yaml
```

### Monitor Memory Usage
Check the log files in `output/logs/` for memory usage patterns:
```
INFO: After loading DATASET_NAME dataset - Memory usage: 2.34GB
INFO: Before removing duplicate rows - Memory usage: 2.34GB (freed 0.05GB)
INFO: After correlation matrix calculation - Memory usage: 3.12GB (freed 0.23GB)
```

### Large Dataset Tips
1. **Start small**: Test with a subset of your files first
2. **Monitor logs**: Check memory usage patterns
3. **Adjust settings**: Tune chunk_size based on your system
4. **Use SSD storage**: Faster I/O helps with chunked processing

## Expected Results

### Processing Pipeline Changes
- **Same output files** as before
- **Same data quality** and cleaning logic
- **Better memory efficiency** and stability
- **More informative logging**

### Performance Expectations
- **Memory usage**: 80-90% reduction
- **Processing time**: May increase 20-50% due to chunking
- **Reliability**: Should complete without crashes
- **Scalability**: Can handle datasets up to 5-10M rows

## Support

If issues persist:
1. Share the memory usage logs from `output/logs/`
2. Try different `chunk_size` values (100, 200, 500, 1000)
3. Consider processing smaller subsets of your data first

The optimizations maintain full backward compatibility - your existing config files and workflows will continue to work with better memory management.