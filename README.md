# Deep Learning Experiments

This repository contains a series of deep learning experiments and a detailed report. The experiments explore fundamental concepts such as model architecture, optimization, loss landscapes, and generalization in neural networks.

## How to Generate Results

### Setting Up the Environment
key tool versions
```plaintext
Python version: 3.12.11
PyTorch version: 2.8.0+cu128
CUDA version: 12.8
Is CUDA available: True
```

### Generating Results
To reproduce the results for each part of the report, run the corresponding script. Each script will generate a new timestamped folder in the `results/` directory containing the plots and data for that specific run.

1. **Part 1: Model Structure Comparison**
   ```bash
   python hw1_1/task_1_1.py
   ```

2. **Part 2: Optimization and Loss Landscape Analysis**
   ```bash
   python hw1_2/task_2_1.py
   ```

3. **Part 3: Fitting Random Labels**
   ```bash
   python hw1_3/task_3_1.py
   ```

4. **Part 4: Model Capacity vs. Performance**
   ```bash
   python hw1_3/task_3_2.py
   ```

5. **Part 5: Loss Landscape Interpolation**
   ```bash
   python hw1_3/task_3_4.py
   ```

6. **Part 6: Batch Size, Generalization, and Sensitivity**
   ```bash
   python hw1_3/task_3_5.py
   ```

### Viewing the Report
The report is available in two formats:
1. **Markdown**: `report_1.md`

## Results Directory
All generated results (plots, logs, and trained model weights) are saved in the `results/` directory. Each script execution creates a new timestamped subdirectory.

## License
This project is open source and licensed under the MIT License.
