import numpy as np
import json


def post_process(
    dicts,
    up,
    OUT_DIR,
    val_name,
    data_type="colmap",
    folder_name="file_path",
    no_post_processing=False,
):
    nframes = len(dicts["frames"])
    if not no_post_processing:
        print("pose processing")
        up = up / np.linalg.norm(up)
        print("up vector was", up)

        # rotate up vector to [0,0,1]
        R = rotmat(up, [0, 0, 1])
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in dicts["frames"]:
            # rotate up to be the z axis
            f["transform_matrix"] = np.matmul(np.eye(4), f["transform_matrix"])

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in dicts["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in dicts["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.01:
                    totp += p * w
                    totw += w
        totp /= totw

        avglen = 0.0
        for f in dicts["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)

        # [Xi] why this ?
        for f in dicts["frames"]:
            f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"
        # l_final_poses.append(f["transform_matrix"])
    for f in dicts["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    print(dicts["frames"][0]["transform_matrix"])
    print(nframes, "frames")

    print(f"writing {OUT_DIR} in format [colmap]")

    if data_type == "colmap":
        with open(OUT_DIR, "w") as outfile:
            json.dump(dicts, outfile, indent=2)
        return

    val_frames_dicts = []
    train_frames_dicts = []

    for idx, elem in enumerate(dicts["frames"]):
        if val_name in elem[folder_name]:
            val_frames_dicts.append(elem)
        else:
            train_frames_dicts.append(elem)

    dicts_train = dict(dicts)
    dicts_val = dict(dicts)

    dicts_val["frames"] = val_frames_dicts
    dicts_train["frames"] = train_frames_dicts

    print(f"writing {OUT_DIR} in format [train_val_test]")
    with open(OUT_DIR.replace(".json", "_train.json"), "w") as outfile:
        json.dump(dicts_train, outfile, indent=2)
    with open(OUT_DIR.replace(".json", "_val.json"), "w") as outfile:
        json.dump(dicts_val, outfile, indent=2)
    with open(OUT_DIR.replace(".json", "_test.json"), "w") as outfile:
        json.dump(dicts_val, outfile, indent=2)
    return


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db):
    """
    returns point closest to both rays of form o+t*d, and a weight factor that goes to 0
    if the lines are parallel.
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f
# /include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose
