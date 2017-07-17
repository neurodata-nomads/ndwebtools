import requests

from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse, FileResponse
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.contrib import messages

import time

from .forms import CutoutForm

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#import png
#from PIL import Image

import tifffile as tiff
import zipfile
import os

import blosc

import re




# All the actual views:

def index(request):
    if request.user.is_authenticated():
        username = get_username(request)
    else:
        username = ''
    request.session['next'] = 'synaptogram:coll_list'
    return render(request,'synaptogram/index.html',{'username': username})

@login_required
def coll_list(request):
    add_url = 'collection/'
    url, headers = get_boss_request(request,add_url)
    resp = get_resp_from_boss(request,url, headers)
    request.session['next'] = 'synaptogram:coll_list'
    if resp == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    collections = resp['collections']
    username = get_username(request)
    context = {'collections': collections, 'username': username}
    return render(request, 'synaptogram/coll_list.html',context)

@login_required
def exp_list(request,coll):
    add_url = 'collection/' + coll + '/experiment/'
    url, headers = get_boss_request(request,add_url)
    resp = get_resp_from_boss(request,url, headers)
    request.session['next'] = '/exp_list/' + coll
    if resp == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    experiments = resp['experiments']

    username = get_username(request)
    context = {'coll': coll, 'experiments': experiments, 'username': username}
    return render(request, 'synaptogram/exp_list.html',context)

@login_required
def cutout(request,coll,exp):
    #we need the channels to fill the form
    channels = get_all_channels(request,coll,exp)
    if channels is None:
        return redirect( reverse('synaptogram:exp_list',args={coll}) )
    elif channels == 'authentication failure':
        request.session['next'] = '/cutout/' + coll + '/' + exp
        return redirect('/openid/openid/KeyCloak')

    #getting the coordinate frame limits for the experiment:
    coord_frame = get_coordinate_frame(request,coll,exp)
    if coord_frame == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    #important stuff out of coord_frame:
        # "x_start": 0,
        # "x_stop": 1000,
        # "y_start": 0,
        # "y_stop": 1000,
        # "z_start": 0,
        # "z_stop": 500    

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:

        form = CutoutForm(request.POST, channels=channels, limits = coord_frame)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            x = str(form.cleaned_data['x_min']) + ':' + str(form.cleaned_data['x_max'])
            y = str(form.cleaned_data['y_min']) + ':' + str(form.cleaned_data['y_max'])
            z = str(form.cleaned_data['z_min']) + ':' + str(form.cleaned_data['z_max'])

            channels = form.cleaned_data['channels']
            
            pass_params_d = {'coll': coll, 'exp': exp,'x': x,'y': y,'z': z, 'channels': ','.join(channels)}
            pass_params = '&'.join(['%s=%s' % (key, value) for (key, value) in pass_params_d.items()])
            params = '?' + pass_params

            # redirect to a new URL:
            end_path = form.cleaned_data['endpoint']
            if end_path == 'sgram':
                return HttpResponseRedirect(reverse('synaptogram:sgram') + params)
            elif end_path == 'cut_urls':
                return HttpResponseRedirect(reverse('synaptogram:cut_url_list') + params)
            elif end_path == 'ndviz':
                return HttpResponseRedirect(reverse('synaptogram:ndviz_url_list') + params)
            elif end_path == 'tiff_stack':
                return HttpResponseRedirect(reverse('synaptogram:tiff_stack') + params)

    # if a GET (or any other method) we'll create a blank form
    else:
        q = request.GET
        x_param = q.get('x')
        if x_param is not None:
            x,y,z = xyz_from_params(q)
            x_rng,y_rng,z_rng = create_voxel_rng(x,y,z)
            form = CutoutForm(channels=channels, 
                initial={   'x_min':str(x_rng[0]),'y_min':str(y_rng[0]),'z_min':str(z_rng[0]),
                            'x_max':str(x_rng[1]),'y_max':str(y_rng[1]),'z_max':str(z_rng[1])},
                limits=coord_frame)
        else:
            form = CutoutForm(channels = channels,
                limits=coord_frame)
    username = get_username(request)
    base_url, headers = get_boss_request(request,'')
    for ch in channels:
        ch_metadata=get_ch_metadata(request,coll,exp,ch)
        if ch_metadata['type'] != 'annotation':
            break
    ndviz_url = ret_ndviz_urls(request,coord_frame,base_url,coll,exp,[ch])[0][0]
    context = {'form': form, 'coll': coll, 'exp': exp, 'username': username, 'ndviz_url':ndviz_url, 'coord_frame': sorted(coord_frame.items())}
    return render(request, 'synaptogram/cutout.html', context)

@login_required
def cut_url_list(request):
    q = request.GET
    coll,exp,x,y,z,channels = process_params(q)
    
    base_url, headers = get_boss_request(request,'')
    urls = ret_cut_urls(request,base_url,coll,exp,x,y,z,channels)
    
    channel_cut_list = zip(channels, urls)

    context = {'channel_cut_list': channel_cut_list, 'coll':coll, 'exp':exp}
    return render(request, 'synaptogram/cut_url_list.html',context)

@login_required
def ndviz_url_list(request):
    q = request.GET
    coll,exp,x,y,z,channels = process_params(q)
    base_url, headers = get_boss_request(request,'')
    coord_frame = get_coordinate_frame(request,coll,exp)
    urls, channels_urls, channels_z = ret_ndviz_urls(request,coord_frame,base_url,coll,exp,channels,x,y,z)

    channel_ndviz_list = zip(channels_urls, channels_z, urls)
    context = {'channel_ndviz_list': channel_ndviz_list, 'coll':coll, 'exp':exp}
    return render(request, 'synaptogram/ndviz_url_list.html',context)

@login_required
def tiff_stack(request):
    q = request.GET
    coll,exp,x,y,z,channels = process_params(q)

    urls=[]
    for ch in channels:
        #create links to go to a method that will download the TIFF images inside each channel
        urls.append(reverse('synaptogram:tiff_stack_channel',args=(coll,exp,x,y,z,ch) ))

    #or package the images and create links for the images
    channels_arg = ','.join(channels)

    return render(request, 'synaptogram/tiff_url_list.html',{'urls': urls, 
        'coll':coll, 'exp':exp, 'x':x, 'y':y, 'z':z, 'channels':channels_arg})

@login_required
def tiff_stack_channel(request,coll,exp,x,y,z,channel):
    fn = create_tiff_stack(request,coll,exp,x,y,z,channel)
    if fn == 'authentication failure' or fn == 'incorrect cutout arguments':
        return redirect( reverse('synaptogram:cutout',args=(coll,exp)))

    serve_data = open(fn, "rb").read()
    response=HttpResponse(serve_data, content_type="image/tiff")
    response['Content-Disposition'] = 'attachment; filename="' + fn.strip('media/') + '"'
    return response

@login_required
def sgram(request):
    q = request.GET
    coll,exp,x,y,z,channels = process_params(q)
    return plot_sgram(request,coll,exp,x,y,z,channels)

@login_required
def sgram_from_ndviz(request):
    url = request.GET.get('url')
    coll,exp,x,y,z = parse_ndviz_url(request,url)

    if coll == 'incorrect source':
        return redirect('synaptogram:coll_list')
    elif coll == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    
    #go to form to let user decide what they want to do
    pass_params_d = {'x': x,'y': y,'z': z}
    pass_params = '&'.join(['%s=%s' % (key, value) for (key, value) in pass_params_d.items()])
    params = '?' + pass_params
    return HttpResponseRedirect(reverse('synaptogram:cutout', args=(coll,exp)) + params)
    #redirect('synaptogram:cutout', coll=coll,exp=exp) 

@login_required
def zip_tiff_stacks(request,coll,exp,x,y,z,channels):
    fname = 'media/' + '_'.join((coll,exp,str(x),str(y),str(z))).replace(':','_') + '.zip'
    
    channels = channels.split(',')

    try:
        os.remove(fname)
    except OSError:
        pass

    with zipfile.ZipFile(fname, mode='x', allowZip64=True) as myzip:
        for ch in channels:
            fn = create_tiff_stack(request,coll,exp,x,y,z,ch)

            if fn == 'authentication failure' or fn == 'incorrect cutout arguments':
                return redirect( reverse('synaptogram:cutout',args=(coll,exp)))
            myzip.write(fn)

    response = FileResponse(open(fname, 'rb'))
    response['Content-Disposition'] = 'attachment; filename="' + fname.strip('media/') + '"'
    return response

def set_sess_exp(request):
    id_token = request.session.get('id_token')
    epoch_time_KC = id_token['exp']
    epoch_time_loc = round(time.time()) # + time.timezone
    new_exp_time = epoch_time_KC - epoch_time_loc
    if new_exp_time < 0: #this really shouldn't happen because we expire sessions now
        return 'expire time in past'
    else:
        request.session.set_expiry(new_exp_time)
        return 0

def get_username(request):
    return request.session.get('userinfo')['name']




# All the interaction with the Boss:

def get_boss_request(request,add_url):
    api_key = request.session.get('access_token')
    # api key should always be present because we should be calling get_boss_request from views with @login_required
    boss_url = 'https://api.boss.neurodata.io/v1/'
    headers = {'Authorization': 'Bearer ' + api_key}
    url = boss_url + add_url
    return url, headers

def get_resp_from_boss(request,url, headers):
    r = requests.get(url, headers = headers)
    if r.status_code == 500:
        messages.error(request, 'server error')
        return 'server error'
    try:
        resp = r.json()
        if ('detail' in resp and resp['detail'] == 'Invalid Authorization header. Unable to verify bearer token') \
                    or set_sess_exp(request) == 'expire time in past':
            return 'authentication failure' #no message - we just redirecto to keycloak
        elif 'message' in resp and 'Incorrect cutout arguments' in resp['message']:
            messages.error(request, resp['message'])
            return 'incorrect cutout arguments'
        return resp
    except ValueError:
        #must be blosc data - return it
        return r

def get_all_channels(request,coll,exp):
    add_url = 'collection/' + coll + '/experiment/' + exp + '/channels/'
    url, headers = get_boss_request(request,add_url)
    resp = get_resp_from_boss(request,url, headers)
    if resp == 'authentication failure':
        return resp #need to handle this better
    elif resp['channels'] == []:
        messages.error(request, 'No channels found for experiment: ' + exp)
        return None
    else:
        channels = tuple(resp['channels'])
    return channels

def get_ch_metadata(request,coll,exp,ch):
    add_url = 'collection/' + coll + '/experiment/' + exp + '/channel/' + ch
    url, headers = get_boss_request(request,add_url)
    resp = get_resp_from_boss(request,url, headers)
    if resp == 'authentication failure':
        return resp#need to handle this better
    return resp

def get_exp_metadata(request,coll,exp):
    add_url = 'collection/' + coll + '/experiment/' + exp
    url, headers = get_boss_request(request,add_url)
    resp = get_resp_from_boss(request,url, headers)
    if resp == 'authentication failure':
        return resp#need to handle this better
    return resp

def get_coordinate_frame(request,coll,exp):
    # https://api.theboss.io/v1/coord/:coordinate_frame
    exp_meta = get_exp_metadata(request,coll,exp)
    if exp_meta == 'authentication failure':
        return exp_meta
    coord_frame = exp_meta['coord_frame']
    add_url = 'coord/' + coord_frame
    url, headers = get_boss_request(request,add_url)
    resp = get_resp_from_boss(request,url, headers)
    if resp == 'authentication failure':
        return resp#need to handle this better
    #check that it contains these data otherwise raise exception:
    # "x_start": 0,
    # "x_stop": 1000,
    # "y_start": 0,
    # "y_stop": 1000,
    # "z_start": 0,
    # "z_stop": 500,
    # "x_voxel_size": 1.0,
    # "y_voxel_size": 1.0,
    # "z_voxel_size": 1.0,
    # "voxel_unit": "nanometers",
    return resp

def get_chan_img_data(request,cut_url,headers_blosc,coll,exp,channel):
    r=get_resp_from_boss(request,cut_url,headers_blosc)
    if r == 'authentication failure' or r == 'incorrect cutout arguments' or r == 'server error':
        return r
    raw_data = blosc.decompress(r.content)
    ch_metadata = get_ch_metadata(request,coll,exp,channel)

    return np.fromstring(raw_data, dtype=ch_metadata['datatype'])

    







#helper functions which process data from the Boss or don't interact with the Boss:

def get_voxel_size(coord_frame):
    x = coord_frame['x_voxel_size']
    y = coord_frame['y_voxel_size']
    z = coord_frame['z_voxel_size']
    return [x,y,z]

def ret_ndviz_urls(request,coord_frame,base_url,coll,exp,channels,x='0:100',y='0:100',z='0:1'):
    #https://viz-dev.boss.neurodata.io/#!%7B%27layers%27:%7B%27synapsinR_7thA%27:%7B%27type%27:%27image%27_%27source%27:%27boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000%27%7D%7D_%27navigation%27:%7B%27pose%27:%7B%27position%27:%7B%27voxelSize%27:[100_100_70]_%27voxelCoordinates%27:[583.1588134765625_5237.650390625_18.5]%7D%7D_%27zoomFactor%27:15.304857247764861%7D%7D
    #unescaped by: http://www.utilities-online.info/urlencode/
    #https://viz-dev.boss.neurodata.io/#!{'layers':{'synapsinR_7thA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[583.1588134765625_5237.650390625_18.5]}}_'zoomFactor':15.304857247764861}}
    ndviz_urls=[]
    channels_urls=[]
    channels_z=[]
    ndviz_base = 'https://viz-dev.boss.neurodata.io/'
    boss_url = 'https://api.boss.neurodata.io/'
    #error check

    xyz_voxel_size = get_voxel_size(coord_frame)
    
    for ch in channels:
        z_rng = list(map(int,z.split(':')))
        ch_metadata = get_ch_metadata(request,coll,exp,ch)
        if ch_metadata['datatype'] == 'uint16':
            window='?window=0,10000'
        else:
            window=''
        for z_val in range(z_rng[0],z_rng[1]):
            ndviz_urls.append(ndviz_base + '#!{\'layers\':{\'' + ch + 
            '\':{\'type\':\'image\'_\'source\':\'boss://' + boss_url
            + coll +'/'+ exp + '/' + ch + window +
            '\'}}_\'navigation\':{\'pose\':{\'position\':{\'voxelSize\':[' + 
            '_'.join(map(str,xyz_voxel_size))
            + ']_\'voxelCoordinates\':[' +
            str(round((sum(list(map(int,x.split(':'))))-1)/2)) + '_' + 
            str(round((sum(list(map(int,y.split(':'))))-1)/2)) + '_' + str(z_val)
            + ']}}_\'zoomFactor\':20}}')
            channels_urls.append(ch)
            channels_z.append(str(z_val))
    return ndviz_urls, channels_urls, channels_z

def ret_cut_urls(request,base_url,coll,exp,x,y,z,channels):
    res=0
    cut_urls=[]
    for ch in channels:
        JJ='/'.join( ('cutout',coll,exp,ch,str(res),x,y,z ) )
        ch_metadata = get_ch_metadata(request,coll,exp,ch)
        if ch_metadata['datatype'] == 'uint16':
            window='?window=0,10000'
        else:
            window=''
        cut_urls.append(base_url + JJ + '/' + window)
    return cut_urls

def error_check_int_param(vals):
    split_val=vals.split(':')
    try:
        val_chk_list = [str(int(a)) for a in split_val]
        vals_chk = ':'.join(val_chk_list)

        #check here if value is within range of the coord_frame, otherwise, raise an exception

        return vals_chk

    except Exception as e:
        print(e)

def xyz_from_params(q):
    x = error_check_int_param(q.get('x'))
    y = error_check_int_param(q.get('y'))
    z = error_check_int_param(q.get('z'))
    return x,y,z

def process_params(q):
    #validation / data sanitization needed here because it's not being done in the form
    # x_rng_str = q.get('x')

    coll = q.get('coll')
    exp = q.get('exp')
    channels = q.get('channels')
    channels = channels.split(',')
    
    x,y,z = xyz_from_params(q)

    return coll,exp,x,y,z,channels

def create_voxel_rng(x,y,z):
    x_rng = list(map(int,x.split(':')))
    y_rng = list(map(int,y.split(':')))
    z_rng = list(map(int,z.split(':')))
    return x_rng,y_rng,z_rng

def create_tiff_stack(request,coll,exp,x,y,z,channel):
    base_url, headers = get_boss_request(request,'')
    cu = ret_cut_urls(request,base_url,coll,exp,x,y,z,[channel])[0]

    x_rng,y_rng,z_rng = create_voxel_rng(x,y,z)

    headers_blosc = headers
    headers_blosc['Content-Type']='application/blosc-python'

    data_mat = get_chan_img_data(request,cu,headers_blosc,coll,exp,channel)

    if data_mat == 'authentication failure' or data_mat == 'incorrect cutout arguments':
        return data_mat

    #if this is a time series, you need to reshape it differently
    img_data = np.reshape(data_mat,
                                (z_rng[1] - z_rng[0],
                                y_rng[1] - y_rng[0],
                                x_rng[1] - x_rng[0]),
                                order='C')
        
    fname = 'media/' + '_'.join((coll,exp,str(x),str(y),str(z),channel)).replace(':','_') + '.tiff'
    tiff.imsave(fname, img_data)
    
    #running out of memory so I am not doing this anymore:
    # image = tiff.imread(fname)
    # np.testing.assert_array_equal(image, img_data)

    return fname

def plot_sgram(request,coll,exp,x,y,z,channels):
    base_url, headers = get_boss_request(request,'')
    cut_urls = ret_cut_urls(request,base_url,coll,exp,x,y,z,channels)
    
    x_rng,y_rng,z_rng = create_voxel_rng(x,y,z)

    num_ch = len(channels)
    num_z = z_rng[1] - z_rng[0]

    headers_blosc = headers
    headers_blosc['Content-Type']='application/blosc-python'

    fig=Figure(figsize=(10, 25), dpi= 150, facecolor='w', edgecolor='k')
    #fig=plt.figure(figsize=(10, 25), dpi= 150, facecolor='w', edgecolor='k')
    for ch_idx,cu in enumerate(cut_urls): #, exception_handler=exception_handler
        data_mat = get_chan_img_data(request,cu,headers_blosc,coll,exp,channels[ch_idx])
        #if this is a time series, you need to reshape it differently
        data_mat_reshape = np.reshape(data_mat,
                                    (z_rng[1] - z_rng[0],
                                    y_rng[1] - y_rng[0],
                                    x_rng[1] - x_rng[0]),
                                    order='C')

        #loop over z and plot them across
        for z_idx in range(data_mat_reshape.shape[0]):
            B=data_mat_reshape[z_idx,:,:]
            ax = fig.add_subplot(num_ch,num_z+1,(num_z+1)*(ch_idx) + (z_idx+1))
            #plt.subplot(num_ch,num_z+1,(num_z+1)*(ch_idx) + (z_idx+1))
            ax.imshow(B, cmap='gray')
            #ax.xticks([])
            #ax.yticks([])
            #if ch_idx is 0:
                #ax.title('z='+ str(z_idx + z_rng[0]))
            #if z_idx is 0:
                #ax.ylabel(channels[ch_idx])
            if z_idx == data_mat_reshape.shape[0]-1:
                C = np.concatenate(data_mat_reshape,axis=1)
                Csum = np.mean(C,axis=1) / 10e3

                ax1 = fig.add_subplot(num_ch,num_z+1,(num_z+1)*(ch_idx) + (z_idx+1) + 1)
                y_idx=np.flip(np.arange(len(Csum)),0)*.8
                ax1.barh(y_idx,Csum,facecolor='blue')
                #ax1.ylim((min(y_idx), max(y_idx)))
                #plt.xlim(0,1)
                #ax.xticks([])
                #ax.yticks([])
    fig.tight_layout(pad=0, rect=[.02, .02, .98, .98] )
    
    # plt.savefig('synaptogram.png')
    # return render(request, 'synaptogram/sgram.html',context)
    canvas=FigureCanvas(fig)
    response=HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

def parse_ndviz_url(request,url):
    #example URL:
    #"https://viz-dev.boss.neurodata.io/#!{'layers':{'CR1_2ndA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/CR1_2ndA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[657.4783325195312_1069.4876708984375_11]}}_'zoomFactor':69.80685914923684}}"
    split_url = url.split('/')
    if split_url[2] != 'viz-dev.boss.neurodata.io':
        return 'incorrect source', None, None, None, None
    coll = split_url[8]
    exp = split_url[9]
    params = split_url[10]

    #incorporate the zoom factor when generating synaptogram from bookmarklet
    #not currently implemented
    match_zoom = re.search(r"(?<=zoomFactor':).*?(?=})",params)
    zoom = int(float(match_zoom.group()))

    match_xyz_voxel = re.search(r"(?<=voxelSize':\[).*?(?=\])",params)
    xyz_voxel = match_xyz_voxel.group()
    xyz_voxel_float = xyz_voxel.split('_')

    match_xyz = re.search(r"(?<=voxelCoordinates':\[).*?(?=\]}}_'zoom)",params)
    xyz = match_xyz.group()
    xyz_float = xyz.split('_')
    xyz_int = [int(float(p)) for p in xyz_float]


    coord_frame = get_coordinate_frame(request,coll,exp)
    if coord_frame == 'authentication failure':
        return coord_frame, None, None, None, None
    #convert the units from ndviz to boss units
    xyz_int = ndviz_units_to_boss(coord_frame, xyz_voxel_float, xyz_int)

    #creates the string param that using now - these will be integer lists at some point
    x,y,z = [(str(p-5) + ':' + str(p+5)) for p in xyz_int]

    return coll,exp,x,y,z

#we need to do this in case the user specified wrong voxel units in the ndviz url
def ndviz_units_to_boss(coord_frame,ndviz_voxel,xyz_int):
    #z doesn't change only xy
    z = xyz_int[2]
    xy  = xyz_int[0:2]
    #doesn't account for time

    boss_vox_size_xy = [coord_frame['x_voxel_size'], coord_frame['y_voxel_size']]
    ndviz_voxel_xy = map(float,ndviz_voxel[0:2])
    
    xy_conv = list(map(lambda a,b,n: round( a/n*b ), xy, boss_vox_size_xy, ndviz_voxel_xy))
    xy_conv.append(z)
    
    return xy_conv