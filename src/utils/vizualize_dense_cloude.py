import open3d as o3d



def vizualize_dense_cloude(point_cloud_path:str):

    dense_pcd = o3d.io.read_point_cloud(
        point_cloud_path
    )


    print(dense_pcd)
    print("Dense points:", len(dense_pcd.points))

    o3d.visualization.draw_geometries([dense_pcd])





