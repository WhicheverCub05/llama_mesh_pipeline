import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

logger = logging.getLogger("llm_pipeline.visual")

def render_obj_to_image(input_dir, output_dir):
    """
    Reads OBJ files from input_dir, parses vertices and faces, and saves 2D plots as .jpg.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    obj_files = list(input_path.glob("*.obj"))
    if not obj_files:
        logger.info(f"No .obj files found in {input_dir}")
        return

    logger.info(f"Found {len(obj_files)} OBJ files. Starting rendering...")

    for obj_file in obj_files:
        try:
            vertices = []
            edges = []
            with open(obj_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    elif line.startswith('f '):
                        parts = line.split()
                        if len(parts) >= 4:
                            # Face vertices (assuming 1-based indexing in OBJ)
                            # parts[1], parts[2], parts[3] are the vertices of the face
                            v_indices = []
                            for i in range(1, 4):
                                # Handle potential v/f format like '1/u/vt'
                                idx = int(parts[i].split('/')[0]) - 1
                                v_indices.append(idx)

                            # Add edges for this face (triangle)
                            edges.append((v_indices[0], v_indices[1]))
                            edges.append((v_indices[1], v_indices[2]))
                            edges.append((v_indices[2], v_indices[0]))

            if not vertices:
                logger.warning(f"No vertices found in {obj_file.name}. Skipping.")
                continue

            # Extract X, Y, Z
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            zs = [v[2] for v in vertices]

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the vertices
            ax.scatter(xs, ys, zs, c='blue', marker='o', s=10)

            # Plot the edges
            for edge in edges:
                if edge[0] < len(vertices) and edge[1] < len(vertices):
                    v1 = vertices[edge[0]]
                    v2 = vertices[edge[1]]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color='black', linewidth=0.5)

            # Set fixed camera view as requested (looking at center of 0-6 moc 64 space)
            ax.set_xlim(0, 64)
            ax.set_ylim(0, 64)
            ax.set_zlim(0, 64)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set camera position
            ax.view_init(elev=30, azim=45)

            output_file = output_path / f"{obj_file.stem}_img.jpg"
            plt.savefig(output_file)
            plt.close(fig)
            logger.info(f"Saved visualization: {output_file}")

        except Exception as e:
            logger.error(f"Failed to render {obj_file.name}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    render_obj_to_image(args.input_dir, args.output_dir)