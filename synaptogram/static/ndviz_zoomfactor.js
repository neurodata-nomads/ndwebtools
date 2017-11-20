function replace_zoomfactor(url, x, voxel_sizes){
    var window_width = window.innerHeight;
    var zoom = voxel_sizes[0] * (x[1] - x[0]) / window_width;
    var url1 = url.replace("'zoomFactor':","'zoomFactor':".concat(String(zoom)));
    return(url1);
}