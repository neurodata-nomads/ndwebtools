import os
import re
import time
import zipfile
from io import BytesIO
from io import StringIO


import blosc
import matplotlib
import numpy as np
import requests
import tifffile as tiff
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views import generic
from django.contrib.auth import user_logged_in
from django.dispatch import receiver

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from .boss_remote import BossRemote
from .block_compute import block_compute
from intern.resource.boss.resource import *
from .forms import CutoutForm, AvatrPullForm, AvatrPushForm

import sys
sys.path.append('synaptogram/')
from NDResource import NeuroDataResource
from block_compute import block_compute
import configparser

# All the actual views:


def index(request):
    request.session['next'] = 'synaptogram:coll_list'
    return render(request, 'synaptogram/index.html')


@receiver(user_logged_in)
def start_login_events(sender, user, request, **kwargs):
    setup_boss_remote(request)
    set_sess_exp(request)


def setup_boss_remote(request):
    # this is called when the user logs in and creates a boss remote class for the given user
    if 'boss_remote' not in request.session:
        request.session['boss_remote'] = BossRemote(request)


def set_sess_exp(request):
    # we set the session expiration to match the bearer token expiration
    id_token = request.session.get('id_token')
    if id_token is not None:  # if admin interfact we aren't using KeyCloak
        epoch_time_KC = id_token['exp']
        epoch_time_loc = round(time.time())  # + time.timezone
        new_exp_time = epoch_time_KC - epoch_time_loc
        request.session.set_expiry(new_exp_time)

def avatr_pull(request, coll, exp, ch, x, y, z, res):
    boss_remote = request.session['boss_remote']
    ch_info = boss_remote.get_ch_info(coll, exp, ch)
    dtype = ch_info['datatype']

    fname = 'media/' + '__'.join((coll, exp, ch)) + '.zip'

    with zipfile.ZipFile(fname, mode='x', allowZip64=True) as myzip:
        img_data, cut_url = get_chan_img_data(request, coll, exp, ch, x, y, z, res)
        #if img_data == 'authentication failure' or img_data == 'incorrect cutout arguments':
        #    raise Exception
        fn = 'media/{}__{}__{}.tiff'.format(
            coll, exp, ch).replace(':', '_')
        tiff.imsave(fn, img_data)
        myzip.write(fn, arcname='image.tiff')

        config = configparser.ConfigParser()
        config['METADATA'] = {'collection': coll,
                             'experiment': exp,
                             'channel': ch,
                             'x': x,
                             'y': y,
                             'z': z,
                             'resolution': res,
                             'dtype': dtype
                             }
        with open('media/metadata.cfg', 'w') as configfile:
            config.write(configfile)
        myzip.write('media/metadata.cfg', arcname='metadata.cfg')

    response = FileResponse(open(fname, 'rb'))
    response['Content-Disposition'] = 'attachment; filename="' + \
        fname.strip('media/') + '"'

    try:
        os.remove(fname)
        os.remove('media/metadata.cfg')
        os.remove(fn)
    except OSError:
        print('not removed')
        pass

    print(cut_url)
    return response

def avatr_push(request, file_img, file_meta, channel):
    boss_remote = request.session['boss_remote']
    #get metadata
    meta = file_meta.read().decode('utf-8')
    param_list = meta.split('\n')
    params = {param.split(' = ')[0]:param.split(' = ')[1] for param in param_list[1:-2]}
    #unpack
    coll = params['collection']
    ch = params['channel']
    exp = params['experiment']
    x_rng = [int(i) for i in params['x'].split(':')]
    y_rng = [int(i) for i in params['y'].split(':')]
    z_rng = [int(i) for i in params['z'].split(':')]
    res = params['resolution']
    old_dtype = params['dtype']

    chan_setup = ChannelResource(ch, coll, exp, 'annotation', datatype='uint64')
    #get image data
    img = tiff.imread(file_img).astype(np.uint64)
    #boss_remote.post(push_url, data=img,)
    #return(params)
    #return 'hi'
    ranges = block_compute(*x_rng,*y_rng,*z_rng)
    #chan = boss_remote.get_project(chan_setup)#NANI!
    for some_range in ranges[0:5]:
        x,y,z = some_range
        some_x = [int(i) for i in x]
        some_y = [int(i) for i in y]
        some_z = [int(i) for i in z]
        some_url = '{}/{}/{}/{}/{}:{}/{}:{}/{}:{}'.format(
            coll,exp,channel,res,x[0],x[1],y[0],y[1],z[0],z[1]
        )
        data_cut = np.stack(
            [np.transpose(np.array(
                img[z_idx][some_x[0]:some_x[1], some_y[0]:some_y[1]], dtype='uint64'
            )) for z_idx in range(some_z[0],some_z[1])]
        )
        print('url: ',some_url)
        print('cube shape: ',data_cut.shape)
        boss_remote.post(some_url, np.ascontiguousarray(data_cut), {'Accept': 'application/blosc'})#np.ascontiguousarray(d
        #boss_remote.create_cutout(chan, 0, x, y, z, np.ascontiguousarray(data_cut))
    return "Successfully pushed"

@login_required
def coll_list(request):
    collections = request.session['boss_remote'].get_collections()

    context = {'collections': collections}
    return render(request, 'synaptogram/coll_list.html', context)


@login_required
def exp_list(request, coll):
    experiments = request.session['boss_remote'].get_experiments(coll)
    if experiments is None:
        messages.error(
            request, 'No experiments found for collection: {}'.format(coll))
        return redirect(reverse('synaptogram:coll_list'))

    context = {'coll': coll, 'experiments': experiments}
    return render(request, 'synaptogram/exp_list.html', context)


@login_required
def cutout(request, coll, exp):
    boss_remote = request.session['boss_remote']
    # we need the channels to fill the form
    channels = boss_remote.get_channels(coll, exp)
    if channels is None:
        messages.error(
            request, 'No channels found for experiment: {}'.format(exp))
        return redirect(reverse('synaptogram:exp_list', args={coll}))

    exp_info = boss_remote.get_exp_info(coll, exp)
    res_vals = list(range(exp_info['num_hierarchy_levels']))

    # getting the coordinate frame limits for the experiment:
    coord_frame = boss_remote.get_coordinate_frame(coll, exp, exp_info)
    # important stuff out of coord_frame:
    # "x_start": 0,
    # "x_stop": 1000,
    # "y_start": 0,
    # "y_stop": 1000,
    # "z_start": 0,
    # "z_stop": 500

    # actual metadata in the BOSS for the experiment
    exp_meta_keyvals = boss_remote.get_exp_metadata(coll, exp)

    # merge the coord_frame data and some experiment metadata together
    exp_data = coord_frame
    copy_keys = ['creator', 'description', 'hierarchy_method',
                 'num_hierarchy_levels', 'num_time_samples',
                 'time_step', 'time_step_unit']
    exp_data.update({key: exp_info[key] for key in copy_keys})
    exp_data.update(exp_meta_keyvals)

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:

        form = CutoutForm(request.POST, channels=channels,
                          limits=coord_frame, res_vals=res_vals)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            x = str(form.cleaned_data['x_min']) + \
                ':' + str(form.cleaned_data['x_max'])
            y = str(form.cleaned_data['y_min']) + \
                ':' + str(form.cleaned_data['y_max'])
            z = str(form.cleaned_data['z_min']) + \
                ':' + str(form.cleaned_data['z_max'])

            res = form.cleaned_data['res_select']

            channels = form.cleaned_data['channels']

            pass_params_d = {'coll': coll, 'exp': exp, 'x': x,
                             'y': y, 'z': z, 'channels': ','.join(channels), 'res': res}
            pass_params = '&'.join(['%s=%s' % (key, value)
                                    for (key, value) in pass_params_d.items()])
            params = '?' + pass_params

            # redirect to a new URL:
            end_path = form.cleaned_data['endpoint']
            if end_path == 'sgram':
                return HttpResponseRedirect(
                    reverse('synaptogram:sgram') + params)
            elif end_path == 'ndviz':
                return HttpResponseRedirect(
                    reverse('synaptogram:ndviz_url_list') + params)
            elif end_path == 'tiff_stack':
                return HttpResponseRedirect(
                    reverse('synaptogram:tiff_stack') + params)

    # if a GET (or any other method) we'll create a blank form
    else:
        q = request.GET
        x_param = q.get('x')
        if x_param is not None:
            x, y, z = xyz_from_params(q)
            x_rng, y_rng, z_rng = create_voxel_rng(x, y, z)
            form = CutoutForm(channels=channels,
                              initial={'x_min': str(x_rng[0]), 'y_min': str(y_rng[0]), 'z_min': str(z_rng[0]),
                                       'x_max': str(x_rng[1]), 'y_max': str(y_rng[1]), 'z_max': str(z_rng[1])},
                              limits=coord_frame, res_vals=res_vals)
        else:
            form = CutoutForm(channels=channels,
                              limits=coord_frame, res_vals=res_vals)

    context = {'form': form, 'coll': coll, 'exp': exp, 'channels': channels,
               'exp_data': sorted(exp_data.items())}
    return render(request, 'synaptogram/cutout.html', context)


@login_required
def get_ndviz_url(request, coll, exp, channel):
    # this generates an ndviz url for the channel/experiment/collection as inputs
    boss_remote = request.session['boss_remote']

    if channel is None:
        channels = boss_remote.get_channels(coll, exp)
    else:
        channels = [channel]
    url, _ = ret_ndviz_urls(request, coll, exp, channels)
    return redirect(url[0])


@login_required
def ndviz_url_list(request):
    boss_remote = request.session['boss_remote']
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)

    coord_frame = boss_remote.get_coordinate_frame(coll, exp)
    voxel_sizes = get_voxel_size(coord_frame)
    urls, z_vals = ret_ndviz_urls(
        request, coll, exp, channels, x, y, z, voxel_sizes)

    x_int = list(map(int, x.split(':')))

    channel_ndviz_list = zip(z_vals, urls)
    context = {
        'channel_ndviz_list': channel_ndviz_list,
        'coll': coll,
        'exp': exp,
        'x': x_int,
        'voxel_sizes': voxel_sizes}
    return render(request, 'synaptogram/ndviz_url_list.html', context)


@login_required
def tiff_stack(request):
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)
    res = q.get('res')

    urls = []
    for ch in channels:
        # create links to go to a method that will download the TIFF images
        # inside each channel
        urls.append(reverse('synaptogram:tiff_stack_channel',
                            args=(coll, exp, x, y, z, ch, res)))

    # or package the images and create links for the images
    channels_arg = ','.join(channels)

    return render(request, 'synaptogram/tiff_url_list.html', {'urls': urls,
                                                              'coll': coll, 'exp': exp, 'x': x, 'y': y, 'z': z, 'channels': channels_arg, 'res': res})


@login_required
def tiff_stack_channel(request, coll, exp, x, y, z, channel, res):
    img_data, cut_url = get_chan_img_data(
        request, coll, exp, channel, x, y, z, res)
    obj = BytesIO()
    tiff.imsave(obj, img_data, metadata={'cut_url': cut_url})

    fname = '{}_{}_{}_{}_{}_{}_{}.tiff'.format(
        coll, exp, x, y, z, channel, res).replace(':', '_')
    response = HttpResponse(obj.getvalue(), content_type='image/TIFF')
    response['Content-Disposition'] = 'attachment; filename="' + fname + '"'
    return response


@login_required
def zip_tiff_stacks(request, coll, exp, x, y, z, channels, res):
    fname = 'media/' + \
        '_'.join(
            (coll, exp, str(x), str(y), str(z))).replace(
            ':', '_') + '.zip'

    channels = channels.split(',')

    try:
        os.remove(fname)
    except OSError:
        pass

    with zipfile.ZipFile(fname, mode='x', allowZip64=True) as myzip:
        for ch in channels:
            img_data, cut_url = get_chan_img_data(
                request, coll, exp, ch, x, y, z, res)
            if img_data == 'authentication failure' or img_data == 'incorrect cutout arguments':
                raise Exception
            fn = 'media/{}_{}_{}_{}_{}_{}_{}.tiff'.format(
                coll, exp, x, y, z, ch, res).replace(':', '_')
            tiff.imsave(fn, img_data, metadata={'cut_url': cut_url})

            # running out of memory so I am not doing this anymore:
            # image = tiff.imread(fname)
            # np.testing.assert_array_equal(image, img_data)

            myzip.write(fn, arcname=fn.strip('media/'))

    response = FileResponse(open(fname, 'rb'))
    response['Content-Disposition'] = 'attachment; filename="' + \
        fname.strip('media/') + '"'
    return response


@login_required
def sgram(request):
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)
    return plot_sgram(request, coll, exp, x, y, z, channels)


@login_required
def sgram_from_ndviz(request):
    url = request.GET.get('url')
    coll, exp = parse_ndviz_url(request, url)

    x = ':'.join(request.GET.get('xextent').split(','))
    y = ':'.join(request.GET.get('yextent').split(','))
    coords = request.GET.get('coords').split(',')
    z_int = int(float(coords[-1]))
    z = '{:d}:{:d}'.format(z_int, z_int + 1)

    # go to form to let user decide what they want to do
    pass_params_d = {'x': x, 'y': y, 'z': z}
    pass_params = '&'.join(['%s=%s' % (key, value)
                            for (key, value) in pass_params_d.items()])
    params = '?' + pass_params
    return HttpResponseRedirect(
        reverse('synaptogram:cutout', args=(coll, exp)) + params)
    #redirect('synaptogram:cutout', coll=coll,exp=exp)


@login_required
def channel_detail(request, coll, exp, channel):
    boss_remote = request.session['boss_remote']
    ch_info = boss_remote.get_ch_info(coll, exp, channel)
    ch_perms = boss_remote.get_permissions(coll, exp, channel)
    perm_sets = ch_perms['permission-sets']
    ndviz_url, _ = ret_ndviz_urls(request, coll, exp, [channel])

    #new stuff
    exp_info = boss_remote.get_exp_info(coll, exp)

    #get resolution, x_start, x_stop, ..., metadata
    res_vals = list(range(exp_info['num_hierarchy_levels']))
    coord_frame = boss_remote.get_coordinate_frame(coll, exp, exp_info)
    exp_meta_keyvals = boss_remote.get_exp_metadata(coll, exp)

    #merge coordinate frame data
    exp_data = coord_frame
    copy_keys = ['creator', 'description', 'hierarchy_method',
                 'num_hierarchy_levels', 'num_time_samples',
                 'time_step', 'time_step_unit']
    exp_data.update({key: exp_info[key] for key in copy_keys})
    exp_data.update(exp_meta_keyvals)

    if request.method == 'POST':
        pull_form = AvatrPullForm(request.POST, limits=coord_frame, res_vals=res_vals)
        push_form = AvatrPushForm(request.POST, request.FILES)

        if pull_form.is_valid():
            # process the data in form.cleaned_data as required
            x = str(pull_form.cleaned_data['x_min']) + \
                ':' + str(pull_form.cleaned_data['x_max'])
            y = str(pull_form.cleaned_data['y_min']) + \
                ':' + str(pull_form.cleaned_data['y_max'])
            z = str(pull_form.cleaned_data['z_min']) + \
                ':' + str(pull_form.cleaned_data['z_max'])
            res = pull_form.cleaned_data['res_select']
            pull_response = avatr_pull(request, coll, exp, channel, x, y, z, res)
            return pull_response

        if push_form.is_valid():
            file_img = push_form.cleaned_data['file']
            file_meta = push_form.cleaned_data['file2']
            push_response = avatr_push(request, file_img, file_meta, channel)
            return HttpResponse(str(push_response))
            return HttpResponse('rip')
            return HttpResponseRedirect(
                reverse('synaptogram:channel_detail', args=(coll, exp, channel))
            )
    else:

        q = request.GET
        x_param = q.get('x')
        if x_param is not None:
            x, y, z = xyz_from_params(q)
            x_rng, y_rng, z_rng = create_voxel_rng(x, y, z)
            pull_form = AvatrPullForm(initial={'x_min': str(x_rng[0]), 'y_min': str(y_rng[0]), 'z_min': str(z_rng[0]),
                                       'x_max': str(x_rng[1]), 'y_max': str(y_rng[1]), 'z_max': str(z_rng[1])},
                              limits=coord_frame, res_vals=res_vals)
        else:
            pull_form = AvatrPullForm(limits=coord_frame,
                                    res_vals=res_vals)
        push_form = AvatrPushForm()

    return render(request, 'synaptogram/channel_detail.html',
                  {'coll': coll, 'exp': exp, 'channel': channel, 'channel_props': ch_info, 'permissions': perm_sets,
                   'ndviz_url': ndviz_url[0], 'pull_form':pull_form, 'push_form':push_form})


@login_required
def start_downsample(request, coll, exp, channel):
    boss_remote = request.session['boss_remote']
    boss_remote.start_downsample(coll, exp, channel)
    return HttpResponseRedirect(
        reverse('synaptogram:channel_detail', args=(coll, exp, channel)))


@login_required
def stop_downsample(request, coll, exp, channel):
    boss_remote = request.session['boss_remote']
    boss_remote.stop_downsample(coll, exp, channel)
    return HttpResponseRedirect(
        reverse('synaptogram:channel_detail', args=(coll, exp, channel)))


# helper functions which process data from the Boss or don't interact with
# the Boss:

def adjust_downsample(res, xy):
    if res > 0:
        xy_adjust = []
        for ext in xy:
            ext_int = [int(aa) for aa in ext.split(':')]
            xy_adjust.append(
                ':'.join([str(round(xx / (2**res))) for xx in ext_int]))
        return xy_adjust
    else:
        return xy

# this is questionable where this should go - helper file or file that
# gets stuff from the boss


def ret_cut_urls(coll, exp, x, y, z, channels, res):
    cut_urls = []
    for ch in channels:
        JJ = '/'.join(('cutout', coll, exp, ch, str(res), x, y, z))
        cut_urls.append(JJ)
    return cut_urls


def get_chan_img_data(request, coll, exp, channel, x, y, z, res):
    boss_remote = request.session['boss_remote']

    res = int(res)
    x, y = adjust_downsample(res, [x, y])

    cut_url = ret_cut_urls(coll, exp, x, y, z, [channel], res)[0]
    r = boss_remote.get(cut_url, {'Accept': 'application/blosc'})

    data_decomp = blosc.decompress(r.content)

    ch_info = boss_remote.get_ch_info(coll, exp, channel)
    ch_datatype = ch_info['datatype']
    data_mat = np.fromstring(data_decomp, dtype=ch_datatype)

    x_rng, y_rng, z_rng = create_voxel_rng(x, y, z)
    # if this is a time series, you need to reshape it differently
    img_data = np.reshape(data_mat,
                          (z_rng[1] - z_rng[0],
                           y_rng[1] - y_rng[0],
                           x_rng[1] - x_rng[0]),
                          order='C')

    return img_data, cut_url


def get_voxel_size(coord_frame):
    x = coord_frame['x_voxel_size']
    y = coord_frame['y_voxel_size']
    z = coord_frame['z_voxel_size']
    return [x, y, z]


def ret_ndviz_channel_part(boss_url, ch_metadata, coll, exp, ch, ch_indx=0):
    if ch_metadata['datatype'] == 'uint16':
        window = 'window=0,10000'
    else:
        window = ''
    if ch_metadata['type'] == 'image':
        chan_type = 'image'
    else:
        chan_type = 'segmentation'

    col_idx = str(ch_indx % 7)  # 7 colors supported by ndviz including white
    blend = '_\'blend\':\'additive\''

    visible_option = ''
    if ch_indx > 2:
        visible_option = '_\'visible\':false'

    #{'ch0':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/ailey-dev/Th1eYFP_control_12/ch0?window=0,10000'_'color':2}_'ch1':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/ailey-dev/Th1eYFP_control_12/ch1'_'opacity':0.45_'color':1}}
    ch_link = ''.join(('\'', ch, '\':{\'type\':\'', chan_type, '\'_\'source\':\'boss://',
                       boss_url, coll, '/', exp, '/', ch, '?', window, '\'', blend, '_\'color\':',
                       col_idx, visible_option, '}'))
    return ch_link


def ret_ndviz_urls(request, coll, exp,
                   channels, x=None, y=None, z=None, voxel_sizes=None):
    # https://viz-dev.boss.neurodata.io/#!%7B%27layers%27:%7B%27synapsinR_7thA%27:%7B%27type%27:%27image%27_%27source%27:%27boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000%27%7D%7D_%27navigation%27:%7B%27pose%27:%7B%27position%27:%7B%27voxelSize%27:[100_100_70]_%27voxelCoordinates%27:[583.1588134765625_5237.650390625_18.5]%7D%7D_%27zoomFactor%27:15.304857247764861%7D%7D
    # unescaped by: http://www.utilities-online.info/urlencode/
    # https://viz-dev.boss.neurodata.io/#!{'layers':{'synapsinR_7thA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[583.1588134765625_5237.650390625_18.5]}}_'zoomFactor':15.304857247764861}}
    ndviz_base = 'https://viz.boss.neurodata.io/'
    boss_url = 'https://api.boss.neurodata.io/'

    boss_remote = request.session['boss_remote']

    if z is not None:
        z_rng = list(map(int, z.split(':')))
    else:
        z_rng = [0, 1]

    ndviz_urls = []
    z_vals = []

    ch_viz_links = []
    for ch_indx, ch in enumerate(channels):
        ch_info = boss_remote.get_ch_info(coll, exp, ch)
        ch_link = ret_ndviz_channel_part(
            boss_url, ch_info, coll, exp, ch, ch_indx)
        ch_viz_links.append(ch_link)

    for z_val in range(z_rng[0], z_rng[1]):
        joined_ndviz_url = ''.join(
            (ndviz_base, '#!{\'layers\':{', '_'.join(ch_viz_links), '}'))

        if x is not None and y is not None and voxel_sizes is not None:
            x_vals = list(map(int, x.split(':')))
            x_mid = str(round(sum(x_vals) / 2))
            y_vals = list(map(int, y.split(':')))
            y_mid = str(round(sum(y_vals) / 2))

            # add navigation
            # we add the zoomfactor in the template, where we can get the accurate window width
            joined_ndviz_url = '{}_\'navigation\':{{\'pose\':{{\'position\':{{\'voxelCoordinates\':[{}_{}_{}]}}}}_\'zoomFactor\':}}}}'.format(
                joined_ndviz_url, x_mid, y_mid, z_val)
        else:
            joined_ndviz_url = '{}}}'.format(joined_ndviz_url)

        ndviz_urls.append(joined_ndviz_url)
        z_vals.append(str(z_val))
    return ndviz_urls, z_vals


def error_check_int_param(vals):
    split_val = vals.split(':')
    try:
        val_chk_list = [str(int(a)) for a in split_val]
        vals_chk = ':'.join(val_chk_list)

        # check here if value is within range of the coord_frame, otherwise,
        # raise an exception

        return vals_chk

    except Exception as e:
        print(e)


def xyz_from_params(q):
    x = error_check_int_param(q.get('x'))
    y = error_check_int_param(q.get('y'))
    z = error_check_int_param(q.get('z'))
    return x, y, z


def process_params(q):
    # validation / data sanitization needed here because it's not being done in the form
    # x_rng_str = q.get('x')

    coll = q.get('coll')
    exp = q.get('exp')
    channels = q.get('channels')
    channels = channels.split(',')

    x, y, z = xyz_from_params(q)

    return coll, exp, x, y, z, channels


def create_voxel_rng(x, y, z):
    x_rng = list(map(int, x.split(':')))
    y_rng = list(map(int, y.split(':')))
    z_rng = list(map(int, z.split(':')))
    return x_rng, y_rng, z_rng


def plot_sgram(request, coll, exp, x, y, z, channels):
    num_ch = len(channels)

    fig = Figure(figsize=(10, 25), dpi=150, facecolor='w', edgecolor='k')
    #fig=plt.figure(figsize=(10, 25), dpi= 150, facecolor='w', edgecolor='k')
    for ch_idx, ch in enumerate(
            channels):  # , exception_handler=exception_handler
        data_mat, _ = get_chan_img_data(request, coll, exp, ch, x, y, z, 0)

        # loop over z and plot them across
        for z_idx in range(data_mat.shape[0]):
            B = data_mat[z_idx, :, :]
            num_z = data_mat.shape[2]
            ax = fig.add_subplot(
                num_ch, num_z + 1, (num_z + 1) * (ch_idx) + (z_idx + 1))
            #plt.subplot(num_ch,num_z+1,(num_z+1)*(ch_idx) + (z_idx+1))
            ax.imshow(B, cmap='gray')
            # ax.xticks([])
            # ax.yticks([])
            # if ch_idx is 0:
            #ax.title('z='+ str(z_idx + z_rng[0]))
            # if z_idx is 0:
            # ax.ylabel(channels[ch_idx])
            if z_idx == data_mat.shape[0] - 1:
                C = np.concatenate(data_mat, axis=1)
                Csum = np.mean(C, axis=1) / 10e3

                ax1 = fig.add_subplot(
                    num_ch, num_z + 1, (num_z + 1) * (ch_idx) + (z_idx + 1) + 1)
                y_idx = np.flip(np.arange(len(Csum)), 0) * .8
                ax1.barh(y_idx, Csum, facecolor='blue')
                #ax1.ylim((min(y_idx), max(y_idx)))
                # plt.xlim(0,1)
                # ax.xticks([])
                # ax.yticks([])
    fig.tight_layout(pad=0, rect=[.02, .02, .98, .98])

    # plt.savefig('synaptogram.png')
    # return render(request, 'synaptogram/sgram.html',context)
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response


def parse_ndviz_url(request, url):
    # example URL:
    #"https://viz-dev.boss.neurodata.io/#!{'layers':{'CR1_2ndA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/CR1_2ndA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[657.4783325195312_1069.4876708984375_11]}}_'zoomFactor':69.80685914923684}}"
    split_url = url.split('/')
    if split_url[2] != 'viz-dev.boss.neurodata.io' and split_url[2] != 'viz.boss.neurodata.io':
        return 'incorrect source', None
    coll = split_url[8]
    exp = split_url[9]

    return coll, exp


def ndviz_units_to_boss(coord_frame, ndviz_voxel, xyz_int):
    # z doesn't change only xy
    z = xyz_int[2]
    xy = xyz_int[0:2]
    # doesn't account for time

    boss_vox_size_xy = [
        coord_frame['x_voxel_size'],
        coord_frame['y_voxel_size']]
    ndviz_voxel_xy = map(float, ndviz_voxel[0:2])

    xy_conv = list(map(lambda a, b, n: round(a / n * b),
                       xy, boss_vox_size_xy, ndviz_voxel_xy))
    xy_conv.append(z)

    return xy_conv
