# Quantum Error Correction with Stim - Repository Analysis

## Overview

This repository contains educational materials and implementations for quantum error correction using the Stim library. It appears to be structured as educational lab exercises focused on two main types of quantum error correcting codes: repetition codes and surface codes.

## Repository Structure

```
.
├── README.md                           # Project overview and setup instructions
├── requirements.txt                    # Comprehensive dependency list (283 packages)
├── .gitignore                         # Standard Python gitignore
├── introduction_to_stim/              # Repetition code lab exercises
│   ├── getting_started.ipynb         # Main tutorial notebook (3.1MB)
│   ├── Introduction to Stim Lab.ipynb # Lab exercises notebook (1.5MB)
│   ├── rep_code.py                    # Student implementation template
│   ├── rc_d3_lecture_07.stim         # Empty Stim circuit files
│   ├── rc_d3_lecture_07_modified.stim
│   └── dont_look/                     # Reference solutions
│       ├── correct_rep_code.py        # Complete repetition code implementation
│       ├── correct_part*.stim         # Solution circuit files
│       ├── compare_utils.py           # Utility functions for comparisons
│       └── *.png                      # Generated plots and visualizations
└── surface_code_in_stem/              # Surface code implementation
    ├── surface_code.py                # Student implementation template
    ├── Surface code in Stim.ipynb     # Main surface code notebook (3.8MB)
    └── dont_look/                     # Reference solutions
        ├── correct_surface_code.py    # Complete surface code implementation
        ├── compare_utils (1).py       # Utility functions
        └── *.png                      # Generated visualizations
```

## Core Technologies and Dependencies

### Primary Libraries
- **Stim**: The main quantum circuit simulator for error correction research
- **PyMatching**: Minimum-weight perfect matching decoder for quantum error correction
- **Sinter**: Statistical analysis and plotting utilities for error correction simulations
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization

### Extended Dependencies
The requirements.txt file contains 283 packages, indicating this might be from a broader development environment that includes:
- ROS 2 (Robot Operating System) packages
- PyTorch and machine learning libraries
- Computer vision libraries (OpenCV)
- Various scientific computing packages

## Educational Structure

### 1. Introduction to Stim (`introduction_to_stim/`)

**Main Notebooks:**
- `getting_started.ipynb`: Comprehensive introduction tutorial covering:
  - Basic Stim usage and circuit creation
  - Detector annotations and sampling
  - Error correction with PyMatching
  - Monte Carlo threshold estimation
  - Surface code analysis and 3D visualization
  
- `Introduction to Stim Lab.ipynb`: Structured lab exercises with:
  - Part 1-3: Building Stim circuits step by step
  - Part 4: Implementing repetition code generators
  - Part 5-8: Simulation and threshold analysis
  - Part 9-10: Follow-up questions and review

**Implementation Templates:**
- `rep_code.py`: Skeleton function for students to implement repetition code generation

### 2. Surface Code Implementation (`surface_code_in_stem/`)

**Main Content:**
- `Surface code in Stim.ipynb`: Advanced tutorial covering:
  - Qubit layout and coordinate systems
  - Lattice construction with CX gates
  - Stabilizer measurements
  - Multi-round error correction protocols
  - Error threshold analysis and projections

**Implementation Framework:**
- `surface_code.py`: Comprehensive template with utility functions and skeleton implementations for:
  - Coordinate system management
  - Lattice construction
  - Stabilizer measurements
  - Error correction rounds
  - Circuit generation

## Key Implementation Details

### Repetition Code (`dont_look/correct_rep_code.py`)
```python
def create_rep_code_stim_string(distance, rounds, p):
    # Creates a distance-d repetition code with:
    # - Initialization with noise
    # - CNOT ladder operations
    # - Stabilizer measurements with detectors
    # - Final data measurements
    # - Observable tracking
```

### Surface Code (`dont_look/correct_surface_code.py`)
Complex implementation featuring:
- **Coordinate System**: Data qubits on integer coordinates, measure qubits at half-integer offsets
- **Stabilizers**: X and Z stabilizer measurements with proper qubit ordering
- **Noise Model**: Depolarizing noise on single and two-qubit operations
- **Error Detection**: Comprehensive detector placement for all stabilizer checks
- **Observable Tracking**: Logical operator measurement for error rate analysis

## Educational Approach

### Progressive Complexity
1. **Basic Circuits**: Simple Bell pairs and measurements
2. **Error Detection**: Detector annotations and syndrome extraction
3. **Error Correction**: PyMatching decoder integration
4. **Statistical Analysis**: Threshold estimation with Sinter
5. **Advanced Codes**: Surface code implementation and analysis

### Hands-on Learning
- Students implement key functions themselves
- Reference solutions hidden in `dont_look/` directories
- Interactive visualizations and 3D models
- Real simulation data and threshold analysis

### Assessment Structure
- Incremental exercises building complexity
- Comparison utilities to check student implementations
- Plotting exercises for data analysis skills
- Conceptual questions about error correction

## Research Applications

This codebase enables:
- **Threshold Analysis**: Determining error thresholds for different codes
- **Performance Comparison**: Benchmarking different error correction strategies
- **Noise Model Studies**: Testing various noise assumptions
- **Scaling Analysis**: Understanding resource requirements vs. protection levels

## Technical Highlights

### Advanced Features
- 3D visualization of surface code lattices
- Circuit-level noise modeling
- Minimum-weight perfect matching decoding
- Statistical significance testing
- Error projection for long-term storage

### Code Quality
- Well-documented utility functions
- Comprehensive error handling
- Modular design for extensibility
- Clear separation of student and reference code

## Conclusion

This repository represents a comprehensive educational framework for learning quantum error correction through hands-on implementation. It successfully bridges theoretical concepts with practical simulation, providing students with both the mathematical understanding and computational skills needed for quantum error correction research.

The combination of Stim's efficient simulation capabilities with interactive Jupyter notebooks creates an effective learning environment for understanding one of quantum computing's most critical challenges.