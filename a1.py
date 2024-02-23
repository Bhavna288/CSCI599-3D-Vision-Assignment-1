import trimesh
from loop_subdivision import subdivision_loop
import os
from mesh import Mesh
import numpy as np

if __name__ == '__main__':

    mesh = trimesh.load_mesh('assets/bunny.obj')
    # mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')

    # mesh.export('assets/cube.obj')

    # # TODO: implement your own loop subdivision here
    iterations = 3
    mesh_subdivided = mesh
    for _ in range(iterations):
        mesh_subdivided = subdivision_loop(mesh_subdivided)
    # print the new mesh information and save the mesh
        print(f'Subdivided Mesh Info: {mesh_subdivided}')

    # mesh_subdivided.show()
    mesh_subdivided.export('assets/bunny_subdivided.obj')

    path = "assets/bunny_subdivided.obj"
    target_v = 200
    mesh = Mesh(path)
    mesh_name = os.path.basename(path).split(".")[-2]
    simp_mesh = mesh.qem_decimation(
        target_v=target_v)
    os.makedirs("assets/", exist_ok=True)
    simp_mesh.save_mesh(
        "assets/{}_{}.obj".format(mesh_name, simp_mesh.vertices.shape[0]))

    mesh_decimated = trimesh.load_mesh(
        "assets/{}_{}.obj".format(mesh_name, simp_mesh.vertices.shape[0]))
    print("Simplified mesh Info: ", mesh_decimated)
    mesh_decimated.show()
