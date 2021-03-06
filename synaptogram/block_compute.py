def block_compute(x_start, x_stop,
                  y_start, y_stop,
                  z_start, z_stop,
                  origin=(0, 0, 0),
                  block_size=(512, 512, 16)):
    """
    Get bounding box coordinates (in 3D) of small cutouts to request in
    order to reconstitute a larger cutout.
    Arguments:
        x_start (int): The lower bound of dimension x
        x_stop (int): The upper bound of dimension x
        y_start (int): The lower bound of dimension y
        y_stop (int): The upper bound of dimension y
        z_start (int): The lower bound of dimension z
        z_stop (int): The upper bound of dimension z
    Returns:
        [((x_start, x_stop), (y_start, y_stop), (z_start, z_stop)), ... ]
    """
    # x
    x_bounds = range(origin[0], x_stop + block_size[0], block_size[0])
    x_bounds = [x for x in x_bounds if (x > x_start and x < x_stop)]
    if len(x_bounds) is 0:
        x_slices = [(x_start, x_stop)]
    else:
        x_slices = []
        for start_x in x_bounds[:-1]:
            x_slices.append((start_x, start_x + block_size[0]))
        x_slices.append((x_start, x_bounds[0]))
        x_slices.append((x_bounds[-1], x_stop))

    # y
    y_bounds = range(origin[1], y_stop + block_size[1], block_size[1])
    y_bounds = [y for y in y_bounds if (y > y_start and y < y_stop)]
    if len(y_bounds) is 0:
        y_slices = [(y_start, y_stop)]
    else:
        y_slices = []
        for start_y in y_bounds[:-1]:
            y_slices.append((start_y, start_y + block_size[1]))
        y_slices.append((y_start, y_bounds[0]))
        y_slices.append((y_bounds[-1], y_stop))

    # z
    z_bounds = range(origin[2], z_stop + block_size[2], block_size[2])
    z_bounds = [z for z in z_bounds if (z > z_start and z < z_stop)]
    if len(z_bounds) is 0:
        z_slices = [(z_start, z_stop)]
    else:
        z_slices = []
        for start_z in z_bounds[:-1]:
            z_slices.append((start_z, start_z + block_size[2]))
        z_slices.append((z_start, z_bounds[0]))
        z_slices.append((z_bounds[-1], z_stop))

    # alright, yuck. but now we have {x, y, z}_slices, each of which hold the
    # start- and end-points of each cube-aligned boundary. For instance, if you
    # requested z-slices 4 through 20, it holds [(4, 16), (16, 20)].

    # For my next trick, I'll convert these to a list of:
    # ((x_start, x_stop), (y_start, y_stop), (z_start, z_stop))

    chunks = []
    for x in x_slices:
        for y in y_slices:
            for z in z_slices:
                chunks.append((x, y, z))
    return chunks
