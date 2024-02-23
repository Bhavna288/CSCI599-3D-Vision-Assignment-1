import numpy as np
import scipy as sp
import heapq
import copy
from tqdm import tqdm
from sklearn.preprocessing import normalize


class Mesh:
    def __init__(self, filepath, is_manifold=True):
        """
        Initializes the Mesh object by loading a mesh from a file, computing
        face normals and centers, and if the mesh is manifold, constructing
        additional mesh data structures including vertex normals, adjacency
        mappings, and a Laplacian matrix.
        """
        self.filepath = filepath
        self.vertices, self.faces = self.load_mesh_from_file(filepath)
        self.computing_face_normals()
        self.computing_face_center()

        if is_manifold:
            self.construct_data()  # self.edges, self.ve
            self.computing_vertex_normals()
            self.establish_vertex_adjacency()
            self.map_vertices_to_faces()
            self.create_laplacian_matrix()

    def load_mesh_from_file(self, filepath):
        """
        Loads mesh data from a given file path. The function expects an OBJ
        file format and extracts vertex positions and face vertex indices.
        """

        vertices, faces = [], []
        f = open(filepath)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vertices.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0])
                                   for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [
                    (ind - 1) if (ind >= 0) else (len(vertices) + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vertices = np.asarray(vertices)
        faces = np.asarray(faces, dtype=int)

        assert np.logical_and(faces >= 0, faces < len(vertices)).all()
        return vertices, faces

    def construct_data(self):
        """
        Constructs the edge data structure for the mesh which includes
        mappings from vertices to edges and vice versa, as well as the
        edge-neighbor and side-neighbor information for each edge.
        """

        self.vert = [[] for _ in self.vertices]
        self.vei = [[] for _ in self.vertices]
        edge_nb = []
        sides = []
        edge_to_key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for _, face in enumerate(self.faces):
            faces_edges = []
            for i in range(3):
                curr = (face[i], face[(i + 1) % 3])
                faces_edges.append(curr)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge_to_key:
                    edge_to_key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.vert[edge[0]].append(edges_count)
                    self.vert[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge_to_key[edge]
                edge_nb[edge_key][nb_count[edge_key]
                                  ] = edge_to_key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] +
                                  1] = edge_to_key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge_to_key[edge]
                sides[edge_key][nb_count[edge_key] -
                                2] = nb_count[edge_to_key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] -
                                1] = nb_count[edge_to_key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count

    def computing_face_normals(self):
        """
        Computes the normals for each face in the mesh using the cross product
        of two edges of each face. Also calculates the area of each face.
        """
        edge1 = self.vertices[self.faces[:, 1]] - \
            self.vertices[self.faces[:, 0]]
        edge2 = self.vertices[self.faces[:, 2]] - \
            self.vertices[self.faces[:, 0]]

        # Cross product of edges gives normals
        calculated_normals = np.cross(edge1, edge2)

        # Normalize these normals
        normals_magnitude = np.linalg.norm(
            calculated_normals, axis=1, keepdims=True) + 1e-24  # Avoid division by zero
        normalized_normals = calculated_normals / normals_magnitude

        # Calculate areas of faces based on the length of the cross product
        calculated_areas = 0.5 * np.linalg.norm(calculated_normals, axis=1)

        # Assigning computed normals and areas to class attributes
        self.face_normals = normalized_normals
        self.face_areas = calculated_areas

    def computing_vertex_normals(self):
        """
        Compute normals for all vertices in the mesh. Each vertex normal is obtained by averaging
        the normals of all faces that vertex is part of, followed by normalization to ensure unit length.
        This method employs sparse matrix multiplication for efficient aggregation of face normals.
        """

        # Initialize a zero matrix for vertex normals with shape (number of vertices, 3)
        norms_per_vertex = np.zeros((len(self.vertices), 3))

        # Retrieve the normals for each face already calculated
        normals_of_faces = self.face_normals

        # Creating a mapping from faces to vertices for the construction of a sparse matrix
        # Repeat each face index 3 times
        face_indices = np.repeat(np.arange(len(self.faces)), 3)
        # Flatten the face vertex indices to a single array
        vertex_indices = self.faces.flatten()

        # Create a sparse matrix where each row corresponds to a vertex and each column to a face,
        # filled with 1s at positions where vertex belongs to the face.
        face_to_vertex_matrix = sp.sparse.coo_matrix(
            (np.ones(len(vertex_indices)), (vertex_indices, face_indices)),
            shape=(len(self.vertices), len(self.faces))
        )

        # Use the sparse matrix to sum the normals of adjacent faces per vertex
        aggregated_normals = face_to_vertex_matrix.dot(normals_of_faces)

        # Normalize these summed normals to unit length for each vertex
        normalized_vertex_normals = normalize(aggregated_normals, axis=1)

        # Update the class attribute with the calculated vertex normals
        self.vertex_normals = normalized_vertex_normals

    def computing_face_center(self):
        """Computes the center of each face as the average of its vertex positions."""
        faces = self.faces
        vs = self.vertices
        self.face_center = np.sum(vs[faces], 1) / 3.0

    def create_laplacian_matrix(self):
        """
        Constructs the Laplacian matrix, a fundamental structure in geometric processing that represents 
        the mesh connectivity. This matrix is crucial for various applications such as mesh smoothing,
        shape analysis, and geometric transformations.
        """
        # Access the mesh edges and vertex-edge adjacency list
        mesh_edges = self.edges
        vertex_edge_adj = self.vert

        # Generate adjacency information for vertices
        vertex_adjacency = [np.unique(mesh_edges[adj_edges].flatten())
                            for adj_edges in vertex_edge_adj]
        vertex_adjacency = [np.setdiff1d(adj_v, np.array([v_idx]), assume_unique=True)
                            for v_idx, adj_v in enumerate(vertex_adjacency)]

        # Count of vertices in the mesh
        total_vertices = len(self.vertices)

        # Preparing row and column indices for the sparse matrix
        row_indices = np.hstack([np.full(len(adj_v), v_idx, dtype=np.int64)
                                for v_idx, adj_v in enumerate(vertex_adjacency)])
        col_indices = np.hstack(vertex_adjacency)

        # Values for the Laplacian matrix construction
        values = np.full_like(row_indices, fill_value=-1.0, dtype=np.float32)

        # Create the sparse matrix representing negative adjacency
        neg_adj_matrix = sp.sparse.csr_matrix(
            (values, (row_indices, col_indices)), shape=(total_vertices, total_vertices))

        # Sum each row to get the diagonal values
        diag_values = -neg_adj_matrix.sum(axis=1).A1

        # Construct the diagonal part of the Laplacian matrix
        row_indices_diag = np.arange(total_vertices)
        col_indices_diag = np.arange(total_vertices)
        laplacian_matrix = sp.sparse.csr_matrix(
            (diag_values, (row_indices_diag, col_indices_diag)), shape=(total_vertices, total_vertices))

        # Combine the negative adjacency with the diagonal to form the Laplacian
        self.laplacian_matrix = neg_adj_matrix + laplacian_matrix

    def map_vertices_to_faces(self):
        """
        Creates a mapping from each vertex to the faces that vertex is part of.
        This is useful for finding the adjacent faces to a vertex.
        """
        vert_to_face = [set() for _ in range(len(self.vertices))]
        for i, f in enumerate(self.faces):
            vert_to_face[f[0]].add(i)
            vert_to_face[f[1]].add(i)
            vert_to_face[f[2]].add(i)
        self.vert_to_face = vert_to_face

    def establish_vertex_adjacency(self):
        """
        Builds a bidirectional mapping between vertices to identify adjacent vertices for each vertex.
        This mapping is crucial for operations requiring knowledge of a vertex's immediate neighborhood.
        """
        adjacency_list = [
            set() for _ in self.vertices]  # Using sets to avoid duplicate entries
        for edge in self.edges:
            adjacency_list[edge[0]].add(edge[1])
            adjacency_list[edge[1]].add(edge[0])

        # Convert sets back to lists for consistency
        self.vertex_neighbors = [list(neighbors)
                                 for neighbors in adjacency_list]

        # Construct the adjacency matrix using sparse matrix techniques for efficiency
        adjacency_indices = np.array([(v, n) for v, neighbors in enumerate(
            self.vertex_neighbors) for n in neighbors], dtype=np.int64).T
        adjacency_values = np.ones(len(adjacency_indices[0]), dtype=np.float32)

        # The shape is square (num_vertices x num_vertices) representing all vertices
        adjacency_shape = (len(self.vertices), len(self.vertices))
        self.adjacency_matrix = sp.sparse.csr_matrix(
            (adjacency_values, adjacency_indices), shape=adjacency_shape)

        # Compute the degree (valency) of each vertex as the sum of connections in the adjacency matrix
        self.vertex_degrees = np.array(
            self.adjacency_matrix.sum(axis=1)).flatten()

    def edge_collapse(self, mesh_simplified, vi_0, vi_1, merged_faces, vertex_active, face_active, vertex_mapping, Q_matrices, edges_priority_queue):
        """
        This method performs the collapse of an edge by merging two vertices into one within the given mesh. 
        It updates the mesh's geometry and topology, recalculates error values for affected edges, and ensures the mesh remains consistent.
        """

        # Identify common neighbors to rewire connections post-collapse
        common_neighbors = list(set(mesh_simplified.vertex_neighbors[vi_0]) & set(
            mesh_simplified.vertex_neighbors[vi_1]))
        new_neighbor_set = (set(mesh_simplified.vertex_neighbors[vi_0]) | set(
            mesh_simplified.vertex_neighbors[vi_1])) - {vi_0, vi_1}

        # Update face associations for the remaining vertex and remove references for the collapsed vertex
        mesh_simplified.vert_to_face[vi_0] = (
            mesh_simplified.vert_to_face[vi_0] | mesh_simplified.vert_to_face[vi_1]) - merged_faces
        mesh_simplified.vert_to_face[vi_1].clear()
        for shared_vertex in common_neighbors:
            mesh_simplified.vert_to_face[shared_vertex] -= merged_faces

        # Reassign neighbor relations to reflect the collapsed edge
        mesh_simplified.vertex_neighbors[vi_0] = list(new_neighbor_set)
        for neighbor in mesh_simplified.vertex_neighbors[vi_1]:
            if neighbor != vi_0:
                mesh_simplified.vertex_neighbors[neighbor] = list(
                    (set(mesh_simplified.vertex_neighbors[neighbor]) - {vi_1}) | {vi_0})
        mesh_simplified.vertex_neighbors[vi_1].clear()

        # Mark the collapsed vertex as inactive
        vertex_active[vi_1] = False

        # Update mappings to reflect the vertex merger for future reference
        vertex_mapping[vi_0].update(vertex_mapping[vi_1], {vi_1})
        vertex_mapping[vi_1].clear()

        # Disable faces that are no longer valid after the edge collapse
        face_active[np.array(list(merged_faces)).astype(np.int32)] = False

        # Calculate the new position for the merged vertex
        mesh_simplified.vertices[vi_0] = (
            mesh_simplified.vertices[vi_0] + mesh_simplified.vertices[vi_1]) / 2

        # Recalculate and update the error metrics for edges emanating from the collapsed vertex
        Q0 = Q_matrices[vi_0]
        for neighbor in mesh_simplified.vertex_neighbors[vi_0]:
            midpoint = (
                mesh_simplified.vertices[vi_0] + mesh_simplified.vertices[neighbor]) / 2
            Q1 = Q_matrices[neighbor]
            Q_combined = Q0 + Q1
            extended_midpoint = np.concatenate([midpoint, [1]])

            # Apply a penalty to the error calculation to prioritize certain collapses
            penalty = 1
            error_metric = np.dot(extended_midpoint, np.dot(
                Q_combined, extended_midpoint.T)) * penalty
            heapq.heappush(edges_priority_queue,
                           (error_metric, (vi_0, neighbor)))

    # def edge_collapse(self, mesh_simplified, vi_0, vi_1, merged_faces, vertex_active, face_active, vertex_mapping, Q_matrices, edges_priority_queue):
    #     """
    #     Collapses an edge between two vertices (vi_0 and vi_1) in the mesh by
    #     merging them into a single vertex, adjusting the mesh topology
    #     accordingly, and updating the edge heap with new error metrics.
    #     """

    #     shared_vv = list(set(mesh_simplified.vertex_neighbors[vi_0]).intersection(
    #         set(mesh_simplified.vertex_neighbors[vi_1])))
    #     new_vi_0 = set(mesh_simplified.vertex_neighbors[vi_0]).union(
    #         set(mesh_simplified.vertex_neighbors[vi_1])).difference({vi_0, vi_1})
    #     mesh_simplified.vert_to_face[vi_0] = mesh_simplified.vert_to_face[vi_0].union(
    #         mesh_simplified.vert_to_face[vi_1]).difference(merged_faces)
    #     mesh_simplified.vert_to_face[vi_1] = set()
    #     mesh_simplified.vert_to_face[shared_vv[0]] = mesh_simplified.vert_to_face[shared_vv[0]
    #                                                                               ].difference(merged_faces)
    #     mesh_simplified.vert_to_face[shared_vv[1]] = mesh_simplified.vert_to_face[shared_vv[1]
    #                                                                               ].difference(merged_faces)

    #     mesh_simplified.vertex_neighbors[vi_0] = list(new_vi_0)
    #     for v in mesh_simplified.vertex_neighbors[vi_1]:
    #         if v != vi_0:
    #             mesh_simplified.vertex_neighbors[v] = list(
    #                 set(mesh_simplified.vertex_neighbors[v]).difference({vi_1}).union({vi_0}))
    #     mesh_simplified.vertex_neighbors[vi_1] = []
    #     vertex_active[vi_1] = False

    #     vertex_mapping[vi_0] = vertex_mapping[vi_0].union(vertex_mapping[vi_1])
    #     vertex_mapping[vi_0] = vertex_mapping[vi_0].union({vi_1})
    #     vertex_mapping[vi_1] = set()

    #     face_active[np.array(list(merged_faces)).astype(np.int32)] = False

    #     mesh_simplified.vertices[vi_0] = 0.5 * \
    #         (mesh_simplified.vertices[vi_0] + mesh_simplified.vertices[vi_1])

    #     """ recomputing E """
    #     Q0 = Q_matrices[vi_0]
    #     for vv_i in mesh_simplified.vertex_neighbors[vi_0]:
    #         v_mid = 0.5 * \
    #             (mesh_simplified.vertices[vi_0] +
    #              mesh_simplified.vertices[vv_i])
    #         Q1 = Q_matrices[vv_i]
    #         Q_combined = Q0 + Q1
    #         v4_mid = np.concatenate([v_mid, np.array([1])])

    #         valence_penalty = 1

    #         E_new = np.matmul(v4_mid, np.matmul(
    #             Q_combined, v4_mid.T)) * valence_penalty
    #         heapq.heappush(edges_priority_queue, (E_new, (vi_0, vv_i)))

    @staticmethod
    def rebuild_mesh(mesh_simplified, vertex_active, face_active, vertex_mapping):
        """
        Rebuilds the mesh after decimation, removing collapsed vertices and
        faces, and updating the mesh topology to reflect the simplified mesh.
        """

        face_map = dict(zip(np.arange(len(vertex_active)),
                        np.cumsum(vertex_active)-1))
        mesh_simplified.vertices = mesh_simplified.vertices[vertex_active]

        vertex_dictionary = {}
        for i, vm in enumerate(vertex_mapping):
            for j in vm:
                vertex_dictionary[j] = i

        for i, f in enumerate(mesh_simplified.faces):
            for j in range(3):
                if f[j] in vertex_dictionary:
                    mesh_simplified.faces[i][j] = vertex_dictionary[f[j]]

        mesh_simplified.faces = mesh_simplified.faces[face_active]
        for i, f in enumerate(mesh_simplified.faces):
            for j in range(3):
                mesh_simplified.faces[i][j] = face_map[f[j]]

        mesh_simplified.computing_face_normals()
        mesh_simplified.computing_face_center()
        mesh_simplified.construct_data()
        mesh_simplified.computing_vertex_normals()
        mesh_simplified.establish_vertex_adjacency()
        mesh_simplified.map_vertices_to_faces()

    @staticmethod
    def build_hash(mesh_simplified, vertex_active, vertex_mapping):
        """
        Builds a hash mapping for vertex indices from the original mesh to the
        simplified mesh and vice versa. This is used for tracking the changes
        made during mesh simplification.
        """

        hash_pool = {}
        unhash_pool = {}
        for simple_id, idx in enumerate(np.where(vertex_active)[0]):
            if len(vertex_mapping[idx]) == 0:
                print("Cannot find parent vertex!")
                return
            for org_i in vertex_mapping[idx]:
                hash_pool[org_i] = simple_id
            unhash_pool[simple_id] = list(vertex_mapping[idx])

        """ check """
        vl_sum = 0
        for vl in unhash_pool.values():
            vl_sum += len(vl)

        if (len(set(hash_pool.keys())) != len(vertex_active)) or (vl_sum != len(vertex_active)):
            print("Cannot cover original vertex!")
            return

        hash_pool = sorted(hash_pool.items(), key=lambda x: x[0])
        mesh_simplified.hash_pool = hash_pool
        mesh_simplified.unhash_pool = unhash_pool

    def save_mesh(self, filename):
        """
        Saves the mesh to a file in OBJ format, including writing the vertex
        positions and face vertex indices.
        """

        with open(filename, 'w') as f:
            for vert in self.vertices:
                f.write('v {} {} {}\n'.format(*vert))
            for face in self.faces + 1:  # OBJ is 1-indexed
                f.write('f {} {} {}\n'.format(*face))

    def compute_Q_for_each_vertex(self):
        Q_matrices = [np.zeros((4, 4)) for _ in self.vertices]
        errors = [0 for _ in self.vertices]  # Initial errors for each vertex

        for vertex_index, vertex in enumerate(self.vertices):
            # Faces adjacent to this vertex
            face_indices = self.vert_to_face[vertex_index]
            Q_matrix = np.zeros((4, 4))
            for face_index in face_indices:
                normal = self.face_normals[face_index]  # Normal of the face
                center = self.face_center[face_index]  # Center of the face
                # Distance from the face to the origin
                distance = -np.dot(normal, center)
                # Plane equation coefficients
                plane = np.append(normal, distance)
                # Accumulate the quadric matrix
                Q_matrix += np.outer(plane, plane)

            Q_matrices[vertex_index] = Q_matrix
            # Homogeneous coordinates of the vertex
            vertex_homogeneous = np.append(vertex, 1)
            errors[vertex_index] = np.dot(vertex_homogeneous, np.dot(
                Q_matrix, vertex_homogeneous))  # Error for this vertex

        return Q_matrices, errors

    def qem_decimation(self, target_v, midpoint=False):
        """
        Implements the Quadric Error Metrics (QEM) mesh simplification algorithm. It reduces the number
        of vertices in the mesh to a specified target count by iteratively collapsing edges based on 
        a calculated error metric.
        """
        vs, vert_to_face, fn, fc, edges = self.vertices, self.vert_to_face, self.face_normals, self.face_center, self.edges

        # Quadric matrices for each vertex
        """ 1. computing Q for each vertex """
        Q_matrices, errors = self.compute_Q_for_each_vertex()

        """ 2. computing E for every possible pairs and create heapq """

        # Initialize a priority queue for edges based on their computed E values
        edges_priority_queue = []

        # Iterate through each edge to calculate its E value
        for index, edge in enumerate(edges):
            vertex_start, vertex_end = vs[edge[0]], vs[edge[1]]
            Q0, Q1 = Q_matrices[edge[0]], Q_matrices[edge[1]]
            Q_combined = Q0 + Q1

            # Determine the new vertex position
            if midpoint:
                # If using the midpoint strategy, simply average the start and end vertices
                vertex_new = (vertex_start + vertex_end) / 2
                vertex4_new = np.concatenate([vertex_new, [1]])
            else:
                # Otherwise, use the least squares method to find the optimal position
                Q_least_squares = np.eye(4)
                Q_least_squares[:3] = Q_combined[:3]
                try:
                    Q_least_squares_inv = np.linalg.inv(Q_least_squares)
                    vertex4_new = np.dot(Q_least_squares_inv, np.array(
                        [[0, 0, 0, 1]]).T).flatten()
                except np.linalg.LinAlgError:
                    # Fall back to the midpoint if inversion fails
                    vertex_new = (vertex_start + vertex_end) / 2
                    vertex4_new = np.concatenate([vertex_new, [1]])

            # Apply a penalty for valence
            penalty_for_valence = 1

            # Calculate the new E value and add it to the priority queue
            E_value_new = np.dot(vertex4_new, np.dot(
                Q_combined, vertex4_new.T)) * penalty_for_valence
            heapq.heappush(edges_priority_queue,
                           (E_value_new, (edge[0], edge[1])))

        """ 3. collapse minimum-error vertex """

        # Duplicate the mesh for simplification
        mesh_simplified = copy.deepcopy(self)

        # Initialize masks for vertices and faces
        vertex_active = np.ones(len(mesh_simplified.vertices), dtype=np.bool_)
        face_active = np.ones(len(mesh_simplified.faces), dtype=np.bool_)

        # Track vertex mappings for collapsed vertices
        vertex_mapping = [{i} for i in range(len(mesh_simplified.vertices))]
        progress_bar = tqdm(total=np.sum(vertex_active) -
                            target_v, desc="Simplifying")
        while np.sum(vertex_active) > target_v:
            if not edges_priority_queue:
                print("No more edges can be collapsed!")
                break

            min_error, (vertex_0, vertex_1) = heapq.heappop(
                edges_priority_queue)

            # Skip if either vertex is already inactive
            if not (vertex_active[vertex_0] and vertex_active[vertex_1]):
                continue

            # Collapse edge
            common_neighbors = set(mesh_simplified.vertex_neighbors[vertex_0]).intersection(
                set(mesh_simplified.vertex_neighbors[vertex_1]))
            faces_to_merge = mesh_simplified.vert_to_face[vertex_0].intersection(
                mesh_simplified.vert_to_face[vertex_1])

            # Check for non-manifold edges or boundary conditions
            if len(common_neighbors) != 2 or len(faces_to_merge) != 2:
                continue
            else:
                self.edge_collapse(mesh_simplified, vertex_0, vertex_1, faces_to_merge, vertex_active,
                                   face_active, vertex_mapping, Q_matrices, edges_priority_queue)
                progress_bar.update(1)

        self.rebuild_mesh(mesh_simplified, vertex_active,
                          face_active, vertex_mapping)
        mesh_simplified.simp = True
        self.build_hash(mesh_simplified, vertex_active, vertex_mapping)

        return mesh_simplified

    def loop_subdivision(self, iterations=1):
        for _ in range(iterations):
            # Create structures to identify unique edges and their opposite vertices
            edge_to_vertices = {}  # Map each edge to its original vertices
            vertex_to_edges = {}  # Map each vertex to its connected edges
            edge_to_faces = {}  # Map each edge to the faces it is part of

            # Step 1: Identify unique edges and connected faces
            for face_id, face in enumerate(self.faces):
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                    if edge not in edge_to_vertices:
                        edge_to_vertices[edge] = []
                    edge_to_vertices[edge].append(face_id)
                    if edge[0] not in vertex_to_edges:
                        vertex_to_edges[edge[0]] = []
                    if edge[1] not in vertex_to_edges:
                        vertex_to_edges[edge[1]] = []
                    vertex_to_edges[edge[0]].append(edge)
                    vertex_to_edges[edge[1]].append(edge)
                    if edge not in edge_to_faces:
                        edge_to_faces[edge] = []
                    edge_to_faces[edge].append(face_id)

            # Identify boundary edges
            boundary_edges = {edge for edge,
                              faces in edge_to_faces.items() if len(faces) == 1}

            # Step 2: Calculate odd vertices (new vertices on edges)
            edge_to_new_vertex = {}
            new_vertices = []
            for edge, vertices in edge_to_vertices.items():
                if edge in boundary_edges:
                    # Boundary edge case
                    new_vertex_position = (
                        self.vertices[edge[0]] + self.vertices[edge[1]]) / 2
                else:
                    # Interior edge case
                    # Need to find the opposite vertices to edge vertices
                    faces = edge_to_faces[edge]
                    opposite_vertices = []
                    for face_id in faces:
                        face_vertices = self.faces[face_id]
                        for v in face_vertices:
                            if v not in edge:
                                opposite_vertices.append(v)
                                break
                    assert len(
                        opposite_vertices) == 2, "Should find exactly two opposite vertices for interior edge"
                    new_vertex_position = 3/8 * (self.vertices[edge[0]] + self.vertices[edge[1]]) + 1/8 * (
                        self.vertices[opposite_vertices[0]] + self.vertices[opposite_vertices[1]])
                edge_to_new_vertex[edge] = len(
                    self.vertices) + len(new_vertices)
                new_vertices.append(new_vertex_position)

            # Step 3: Adjust the position of even vertices (original vertices)
            adjusted_vertices = np.copy(self.vertices)
            for vertex_id in range(len(self.vertices)):
                if vertex_id in vertex_to_edges:
                    # Number of adjacent vertices
                    N = len(vertex_to_edges[vertex_id])
                    if any(edge in boundary_edges for edge in vertex_to_edges[vertex_id]):
                        # Boundary vertex case
                        boundary_adjacent_vertices = [vertex for edge in vertex_to_edges[vertex_id]
                                                      if edge in boundary_edges for vertex in edge if vertex != vertex_id]
                        adjusted_vertices[vertex_id] = 1/8 * sum(
                            self.vertices[adj_vertex] for adj_vertex in boundary_adjacent_vertices) + 3/4 * self.vertices[vertex_id]
                    else:
                        # Interior vertex case
                        beta = 1/N * \
                            (5/8 - (3/8 + 1/4 * np.cos(2 * np.pi / N))**2)
                        sum_adjacent = sum(
                            self.vertices[edge[1] if edge[0] == vertex_id else edge[0]] for edge in vertex_to_edges[vertex_id])
                        adjusted_vertices[vertex_id] = (
                            1 - N * beta) * self.vertices[vertex_id] + beta * sum_adjacent

            # Step 4: Compose new faces
            new_faces = []
            new_face_vertices = []  # To store the new vertices created at the center of the old faces
            for face in self.faces:
                # Calculate new face vertex (center of the old face)
                face_center = np.mean([self.vertices[v] for v in face], axis=0)
                face_center_index = len(
                    self.vertices) + len(new_vertices) + len(new_face_vertices)
                new_face_vertices.append(face_center)

                # Indices of the new vertices created at the midpoints of the edges
                edge_vertices = [edge_to_new_vertex[tuple(
                    sorted([face[i], face[(i + 1) % 3]]))] for i in range(3)]

                # Create four new faces for each original face
                for i in range(3):
                    new_faces.append(
                        [face[i], edge_vertices[i], face_center_index, edge_vertices[(i - 1) % 3]])

            # Update mesh with new vertices and faces
            self.vertices = np.verticestack((adjusted_vertices, np.array(
                new_vertices), np.array(new_face_vertices)))
            self.faces = np.array(new_faces)

            # # Rebuild necessary mesh data structures after modification
