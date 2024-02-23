# 3D Mesh Processing Toolkit

This toolkit contains implementations for two fundamental algorithms in 3D mesh processing: Loop Subdivision for mesh refinement and Quadric Error Metrics (QEM) for mesh simplification. This toolkit allows for the enhancement or reduction of mesh detail to fit various application needs in 3D vision.

## Directory Structure

- `a1.py`: Main script that includes functions for performing Loop Subdivision and QEM simplification.
- `loop_subdivision.py`: Contains the Loop Subdivision function which refines and smooths mesh models.
- `mesh.py`: Defines the Mesh class used to construct and manipulate mesh data for QEM simplification.
- `assets/`: Folder containing the generated 3D object files resulting from the subdivision and simplification processes.

## Features

- **Loop Subdivision**: Increases the number of vertices and faces in a mesh, resulting in a smoother approximation of the original geometry using the `trimesh` library.
- **QEM Simplification**: Reduces the complexity of a mesh while minimizing the geometric error, preserving the essential form with a custom mesh data structure.

## Usage

1. Ensure that all dependencies are installed, including `numpy`, `scipy`, and `trimesh`.
2. Place your original `.obj` mesh files in the `assets/` directory.
3. Run the `a1.py` script to perform mesh processing. Choose between Loop Subdivision and QEM simplification based on your requirements.

## Implementation

- The `loop_subdivision` function in `loop_subdivision.py` takes a mesh object and performs subdivision based on the Loop algorithm.
- The `mesh.py` script creates a Mesh class instance, providing the necessary data structures and methods for QEM simplification.

## Results

The `assets/` directory will be populated with the resulting 3D models post-processing. They can be used for visualization, analysis, or as input for further processing in 3D applications. index.html consists the writeup