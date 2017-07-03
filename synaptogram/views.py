#import grequests # for async requests, conflicts with requests
import requests

from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.conf import settings

from .forms import *

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import tifffile as tiff
import zipfile
import os

#import png
#from PIL import Image

import blosc

import re

# Create your views here.

def index(request):
    if request.user.is_authenticated():
        username = get_username(request)
    else:
        username = ''
    return render(request,'synaptogram/index.html',{'username': username})

def get_boss_request(request,add_url):
    api_key = request.session['access_token']
    boss_url = 'https://api.boss.neurodata.io/v1/'
    headers = {'Authorization': 'Bearer ' + api_key}
    url = boss_url + add_url
    return url, headers

def get_resp_from_boss(url, headers):
    r = requests.get(url, headers = headers)
    resp = r.json()
    if 'detail' in resp and resp['detail'] == 'Invalid Authorization header. Unable to verify bearer token':
        return 'error'
    return r

@login_required
def coll_list(request):
    add_url = 'collection/'
    url, headers = get_boss_request(request,add_url)
    r = get_resp_from_boss(url, headers)
    if r == 'error':
        return redirect('/openid/openid/KeyCloak', args={'next':'synaptogram:coll_list'})
    response = r.json()
    collections = response['collections']
    username = get_username(request)
    context = {'collections': collections, 'username': username}
    return render(request, 'synaptogram/coll_list.html',context)

def get_username(request):
    return request.session['userinfo']['name']

@login_required
def exp_list(request,coll):
    add_url = 'collection/' + coll + '/experiment/'
    url, headers = get_boss_request(request,add_url)
    r = get_resp_from_boss(url, headers)
    if r == 'error':
        return redirect('/openid/openid/KeyCloak', args={'next':'synaptogram:exp_list'})
    response = r.json()
    experiments = response['experiments']

    username = get_username(request)
    context = {'coll': coll, 'experiments': experiments, 'username': username}
    return render(request, 'synaptogram/exp_list.html',context)

def get_all_channels(request,coll,exp):
    add_url = 'collection/' + coll + '/experiment/' + exp + '/channels/'
    url, headers = get_boss_request(request,add_url)
    r = get_resp_from_boss(url, headers)
    if r == 'error':
        return [] #need to handle this better
    response = r.json()
    channels = tuple(response['channels'])
    return channels

@login_required
def cutout(request,coll,exp):
    channels = get_all_channels(request,coll,exp)
    
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:

        form = CutoutForm(request.POST, channels=channels)
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
                initial={'x_min':str(x_rng[0]),'y_min':str(y_rng[0]),'z_min':str(z_rng[0]),
                'x_max':str(x_rng[1]),'y_max':str(y_rng[1]),'z_max':str(z_rng[1])})
        else:
            form = CutoutForm(channels = channels)
    username = get_username(request)
    base_url, headers = get_boss_request(request,'')
    ch=channels[0]
    #https://viz-dev.boss.neurodata.io/#!{'layers':{'image':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/ben_dev/sag_left_junk/image'}}_'navigation':{'pose':{'position':{'voxelSize':[4_4_3]_'voxelCoordinates':[1080_1280_1082.5]}}_'zoomFactor':3}}
    ndviz_url = ret_ndviz_urls(base_url,coll,exp,[ch])[0][0]
    context = {'form': form, 'coll': coll, 'exp': exp, 'username': username, 'ndviz_url':ndviz_url}
    return render(request, 'synaptogram/cutout.html', context)

def ret_cut_urls(request,base_url,coll,exp,x,y,z,channels):
    res=0
    cut_urls=[]
    for ch in channels:
        JJ='/'.join( ('cutout',coll,exp,ch,str(res),x,y,z ) )
        dtype = get_ch_dtype(request,coll,exp,ch)
        if dtype is 'uint16':
            window='?window=0,10000'
        else:
            window=''
        cut_urls.append(base_url + JJ + '/' + window)
    return cut_urls

@login_required
def cut_url_list(request):
    coll,exp,x,y,z,channels = process_params(request)
    
    base_url, headers = get_boss_request(request,'')
    urls = ret_cut_urls(request,base_url,coll,exp,x,y,z,channels)
    
    channel_cut_list = zip(channels, urls)

    context = {'channel_cut_list': channel_cut_list}
    return render(request, 'synaptogram/cut_url_list.html',context)

def ret_ndviz_urls(base_url,coll,exp,channels,x='0:100',y='0:100',z='0:1'):
    #https://viz-dev.boss.neurodata.io/#!%7B%27layers%27:%7B%27synapsinR_7thA%27:%7B%27type%27:%27image%27_%27source%27:%27boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000%27%7D%7D_%27navigation%27:%7B%27pose%27:%7B%27position%27:%7B%27voxelSize%27:[100_100_70]_%27voxelCoordinates%27:[583.1588134765625_5237.650390625_18.5]%7D%7D_%27zoomFactor%27:15.304857247764861%7D%7D
    #unescaped by: http://www.utilities-online.info/urlencode/
    #https://viz-dev.boss.neurodata.io/#!{'layers':{'synapsinR_7thA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[583.1588134765625_5237.650390625_18.5]}}_'zoomFactor':15.304857247764861}}
    ndviz_urls=[]
    channels_urls=[]
    channels_z=[]
    ndviz_base = 'https://viz-dev.boss.neurodata.io/'
    boss_url = 'https://api.boss.neurodata.io/'
    
    for ch in channels:
        z_rng = list(map(int,z.split(':')))
        for z_val in range(z_rng[0],z_rng[1]):
            ndviz_urls.append(ndviz_base + '#!{\'layers\':{\'' + ch + 
            '\':{\'type\':\'image\'_\'source\':\'boss://' + boss_url
            + coll +'/'+ exp + '/' + ch + 
            '?window=0,10000\'}}_\'navigation\':{\'pose\':{\'position\':{\'voxelSize\':[100_100_70]_\'voxelCoordinates\':['
            + x.split(':')[0] + '_' + y.split(':')[0] + '_' + str(z_val)
            + ']}}_\'zoomFactor\':20}}')
            channels_urls.append(ch)
            channels_z.append(str(z_val))
    return ndviz_urls, channels_urls, channels_z

def error_check_int_param(vals):
    split_val=vals.split(':')
    try:
        val_chk_list = [str(int(a)) for a in split_val]
        vals_chk = ':'.join(val_chk_list)
        return vals_chk
    except Exception as e:
        print(e)

def xyz_from_params(q):
    x = error_check_int_param(q.get('x'))
    y = error_check_int_param(q.get('y'))
    z = error_check_int_param(q.get('z'))
    return x,y,z

def process_params(request):
    q = request.GET
    
    #validation / data sanitization needed here because it's not being done in the form
    # x_rng_str = q.get('x')

    coll = q.get('coll')
    exp = q.get('exp')
    channels = q.get('channels')
    channels = channels.split(',')
    
    x,y,z = xyz_from_params(q)

    return coll,exp,x,y,z,channels

def ndviz_url_list(request):
    coll,exp,x,y,z,channels = process_params(request)
    base_url, headers = get_boss_request(request,'')
    urls, channels_urls, channels_z = ret_ndviz_urls(base_url,coll,exp,channels,x,y,z)

    channel_ndviz_list = zip(channels_urls, channels_z, urls)
    context = {'channel_ndviz_list': channel_ndviz_list}
    return render(request, 'synaptogram/ndviz_url_list.html',context)

def tiff_stack(request):
    coll,exp,x,y,z,channels = process_params(request)
    base_url, headers = get_boss_request(request,'')

    urls=[]
    for ch in channels:
        #create links to go to a method that will download the TIFF images inside each channel
        urls.append(reverse('synaptogram:tiff_stack_channel',args=(coll,exp,x,y,z,ch) ))

    #or package the images and create links for the images
    channels_arg = ','.join(channels)

    return render(request, 'synaptogram/tiff_url_list.html',{'urls': urls, 
        'coll':coll, 'exp':exp, 'x':x, 'y':y, 'z':z, 'channels':channels_arg})

def create_voxel_rng(x,y,z):
    x_rng = list(map(int,x.split(':')))
    y_rng = list(map(int,y.split(':')))
    z_rng = list(map(int,z.split(':')))
    return x_rng,y_rng,z_rng

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
            myzip.write(fn)

    # zipfile.ZipFile(file, mode='x', compression=ZIP_LZMA, allowZip64=True)

    serve_data = open(fname, "rb").read()
    response=HttpResponse(serve_data, content_type="image/zip")
    response['Content-Disposition'] = 'attachment; filename="' + fname.strip('media/') + '"'
    return response


def create_tiff_stack(request,coll,exp,x,y,z,channel):
    base_url, headers = get_boss_request(request,'')
    cu = ret_cut_urls(request,base_url,coll,exp,x,y,z,[channel])[0]

    x_rng,y_rng,z_rng = create_voxel_rng(x,y,z)

    headers_blosc = headers
    headers_blosc['Content-Type']='application/blosc-python'

    data_mat = get_chan_img_data(request,cu,headers_blosc,coll,exp,channel)
    #if this is a time series, you need to reshape it differently
    img_data = np.reshape(data_mat,
                                (z_rng[1] - z_rng[0],
                                y_rng[1] - y_rng[0],
                                x_rng[1] - x_rng[0]),
                                order='C')
        
    fname = 'media/' + '_'.join((coll,exp,str(x),str(y),str(z),channel)).replace(':','_') + '.tiff'
    tiff.imsave(fname, img_data)
    
    image = tiff.imread(fname)
    np.testing.assert_array_equal(image, img_data)

    return fname

def tiff_stack_channel(request,coll,exp,x,y,z,channel):
    fname = create_tiff_stack(request,coll,exp,x,y,z,channel)
    serve_data = open(fname, "rb").read()
    response=HttpResponse(serve_data, content_type="image/tiff")
    response['Content-Disposition'] = 'attachment; filename="' + fname.strip('media/') + '"'
    return response

@login_required
def sgram(request):
    coll,exp,x,y,z,channels = process_params(request)
    return plot_sgram(request,coll,exp,x,y,z,channels)

def get_ch_dtype(request,coll,exp,ch):
    add_url = 'collection/' + coll + '/experiment/' + exp + '/channel/' + ch
    url, headers = get_boss_request(request,add_url)
    r = get_resp_from_boss(url, headers)
    if r == 'error':
        []#need to handle this better
    response = r.json()
    return response['datatype']

def get_chan_img_data(request,cut_url,headers_blosc,coll,exp,channel):
    r=get_resp_from_boss(url, headers_blosc)
    if r == 'error':
        []#need to handle this better
    raw_data = blosc.decompress(r.content)

    dtype = get_ch_dtype(request,coll,exp,channel)

    return np.fromstring(raw_data, dtype=dtype)

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
            if z_idx is data_mat_reshape.shape[0]-1:
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

def parse_ndviz_url(url):
    #example URL:
    #"https://viz-dev.boss.neurodata.io/#!{'layers':{'CR1_2ndA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/CR1_2ndA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[657.4783325195312_1069.4876708984375_11]}}_'zoomFactor':69.80685914923684}}"
    split_url = url.split('/')
    coll = split_url[8]
    exp = split_url[9]
    params = split_url[10]

    #incorporate the zoom factor when generating synaptogram from bookmarklet
    match_zoom = re.search(r"(?<=zoomFactor':).*?(?=}})",params)
    zoom = int(float(match_zoom.group()))

    # import pdb; pdb.set_trace()

    match_xyz = re.search(r"(?<=voxelCoordinates':\[).*?(?=\]}}_'zoom)",params)
    xyz = match_xyz.group()
    xyz_float = xyz.split('_')
    xyz_int = [int(float(p)) for p in xyz_float]

    #creates the string param that using now - these will be integer lists at some point
    x,y,z = [(str(p-5) + ':' + str(p+5)) for p in xyz_int]

    return coll,exp,x,y,z

@login_required
def sgram_from_ndviz(request):
    url = request.GET.get('url')
    coll,exp,x,y,z = parse_ndviz_url(url)
    #get all the channels
    #channels = get_all_channels(request,coll,exp)
    # return plot_sgram(request,coll,exp,x,y,z,channels)
    
    #go to form to let user decide what they want to do
    pass_params_d = {'x': x,'y': y,'z': z}
    pass_params = '&'.join(['%s=%s' % (key, value) for (key, value) in pass_params_d.items()])
    params = '?' + pass_params
    return HttpResponseRedirect(reverse('synaptogram:cutout', args=(coll,exp)) + params)
    #redirect('synaptogram:cutout', coll=coll,exp=exp) 