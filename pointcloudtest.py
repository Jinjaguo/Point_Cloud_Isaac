import asyncio
import omni.replicator.core as rep
import numpy as np
import open3d 

async def test_pointcloud():
    print("Initializing scene...")

    # Add Light
    distance_light = rep.create.light(rotation=(315, 0, 0), intensity=10000, light_type="distant")

    # Add Cube with semantics
    cube = rep.create.cube(
        position=(0, 0, 0), 
        scale=(1.0, 1.0, 1.0), 
        semantics=[("class", "cube")]
    )

    # Configure Camera
    W, H = (1920, 1080)
    camera = rep.create.camera(
        position=(2.0, 2.0, 2.0), 
        look_at=cube, 
    )
    render_product = rep.create.render_product(camera, (W, H))

    # Attach Pointcloud Annotator
    pointcloud_anno = rep.annotators.get("pointcloud")
    pointcloud_anno.attach(render_product)  # Disable semantic filtering

    # Capture Pointcloud Data
    points = []
    colors = []

    for frame in range(3):
        print(f"Capturing frame {frame + 1}...")
        await rep.orchestrator.step_async()

        pc_data = pointcloud_anno.get_data()

        if pc_data["data"].size == 0:
            print("Empty point data, skipping this frame.")
            continue

        points.append(pc_data["data"])
        colors.append(pc_data["info"]["pointRgb"].reshape(-1, 4)[:, :3] / 255.0)

    # Combine and Save Point Cloud
    if points:
        pc_data_combined = np.concatenate(points)
        pc_colors_combined = np.concatenate(colors)
    else:
        print("No valid point data captured.")
        return
        

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc_data_combined)
    pcd.colors = open3d.utility.Vector3dVector(pc_colors_combined)
    open3d.io.write_point_cloud("pointcloud.ply", pcd)
    print("Point cloud saved to pointcloud.ply")

asyncio.ensure_future(test_pointcloud())

