# 🧠 3D Mesh Data Preprocessing for AI Models

### 📚 Course Assignment
**Topic:** Data Preprocessing for 3D Meshes  
**Student:** Akhil Puri
**Repository:** AssignmentM
**Environment:** Python 3.11.2 + Trimesh, NumPy, Matplotlib

---

## 🚀 Project Overview

This project focuses on building a **data preprocessing pipeline** for 3D mesh data — a crucial step before training AI models that understand 3D geometry (such as SeamGPT).

The tasks include:
1. **Loading & Inspecting 3D Meshes**  
2. **Normalization & Quantization of Mesh Vertices**  
3. **Dequantization, Denormalization & Error Measurement**

No AI training is performed — the focus is on **data cleaning and consistency**.

---

## 📂 Folder Structure


## 🧩 Task 1 — Load and Inspect the Mesh (20 Marks)

**Goal:** Load .OBJ files, visualize basic properties, and inspect mesh structure.  
**Libraries Used:** `trimesh`, `numpy`

**Key Steps:**
- Loaded meshes using `trimesh.load_mesh()`
- Extracted vertices, faces, and normals
- Printed mesh summaries (vertex count, face count)
- Visualized bounding box and centroid

# Min–Max Normalization
norm_minmax = (vertices - v_min) / (v_max - v_min)

# Unit Sphere Normalization
norm_sphere = (vertices - center) / radius
mse = np.mean((orig_vertices - reconstructed) ** 2)
mae = np.mean(np.abs(orig_vertices - reconstructed))

```python
import trimesh
mesh = trimesh.load_mesh("8samples/table.obj", skip_materials=True)
print(mesh)
mesh.show()
import trimesh, numpy as np, os

mesh = trimesh.load_mesh("8samples/table.obj", skip_materials=True, process=False)
vertices = np.array(mesh.vertices)

# --- Min–Max Normalization ---
v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
norm_minmax = (vertices - v_min) / (v_max - v_min)

# --- Unit-Sphere Normalization ---
center = vertices.mean(axis=0)
radius = np.max(np.linalg.norm(vertices - center, axis=1))
norm_sphere = (vertices - center) / radius

# --- Quantization ---
bins = 1024
quant_minmax = np.floor(norm_minmax * (bins - 1)).astype(int)
quant_sphere = np.floor(((norm_sphere + 1) / 2) * (bins - 1)).astype(int)

```
