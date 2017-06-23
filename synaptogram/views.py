#import grequests # for async requests, conflicts with requests
import requests

from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.views import generic
from django.utils import timezone

from .forms import *
from .models import User

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import png
#from PIL import Image

import blosc

import re

from django.conf import settings

# Create your views here.

def index(request):
    return render(request, 'synaptogram/index.html',context=None)

def login(request):
    #users = User.all
    #context=users

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = LoginForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            username = form.cleaned_data['username']
            user = User.objects.get(name=username)

            request.session['api_key']=user.api_key
            
            # redirect to a new URL:
            return HttpResponseRedirect('coll_list')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = LoginForm()

    return render(request, 'synaptogram/login.html', {'form': form})

def sign_up(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = SignupForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            username = form.cleaned_data['username']
            api_key = form.cleaned_data['api_key']
            
            user = User(name=username, api_key=api_key)
            user.save()

            # redirect to a new URL:
            return HttpResponseRedirect('login')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = SignupForm()

    return render(request, 'synaptogram/sign_up.html', {'form': form})

def get_boss_request(request,add_url):
    api_key = request.session['api_key']
    boss_url = 'https://api.boss.neurodata.io/v1/'
    headers = {'Authorization': 'Token ' + api_key}
    url = boss_url + add_url
    return url, headers

def coll_list(request):
    add_url = 'collection/'
    url, headers = get_boss_request(request,add_url)
    r = requests.get(url, headers = headers)
    response = r.json()
    collections = response['collections']

    context = {'collections': collections}
    return render(request, 'synaptogram/coll_list.html',context)

def exp_list(request,coll):
    add_url = 'collection/' + coll + '/experiment/'
    url, headers = get_boss_request(request,add_url)
    r = requests.get(url, headers = headers)
    response = r.json()
    experiments = response['experiments']

    context = {'coll': coll, 'experiments': experiments}
    return render(request, 'synaptogram/exp_list.html',context)

def get_all_channels(request,coll,exp):
    add_url = 'collection/' + coll + '/experiment/' + exp + '/channels/'
    url, headers = get_boss_request(request,add_url)
    r = requests.get(url, headers = headers)
    response = r.json()
    channels = tuple(response['channels'])
    return channels

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
        form = CutoutForm(channels = channels)
    context = {'form': form, 'coll': coll, 'exp': exp}
    return render(request, 'synaptogram/cutout.html', context)

def ret_cut_urls(base_url,coll,exp,x,y,z,channels):
    res=0
    cut_urls=[]
    for ch in channels:
        JJ='/'.join( ('cutout',coll,exp,ch,str(res),x,y,z ) )
        window='?window=0,10000'
        cut_urls.append(base_url + JJ + '/' + window)
    return cut_urls

def cut_url_list(request):
    coll,exp,x,y,z,channels = process_params(request)
    
    base_url, headers = get_boss_request(request,'')
    urls = ret_cut_urls(base_url,coll,exp,x,y,z,channels)
    
    channel_cut_list = zip(channels, urls)

    context = {'channel_cut_list': channel_cut_list}
    return render(request, 'synaptogram/cut_url_list.html',context)

def ret_ndviz_urls(base_url,coll,exp,x,y,z,channels):
    #https://viz-dev.boss.neurodata.io/#!%7B%27layers%27:%7B%27synapsinR_7thA%27:%7B%27type%27:%27image%27_%27source%27:%27boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000%27%7D%7D_%27navigation%27:%7B%27pose%27:%7B%27position%27:%7B%27voxelSize%27:[100_100_70]_%27voxelCoordinates%27:[583.1588134765625_5237.650390625_18.5]%7D%7D_%27zoomFactor%27:15.304857247764861%7D%7D
    #unescaped by: http://www.utilities-online.info/urlencode/
    #https://viz-dev.boss.neurodata.io/#!{'layers':{'synapsinR_7thA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[583.1588134765625_5237.650390625_18.5]}}_'zoomFactor':15.304857247764861}}
    ndviz_urls=[]
    channels_urls=[]
    channels_z=[]
    ndviz_base = 'https://viz-dev.boss.neurodata.io/'
    
    for ch in channels:
        z_rng = list(map(int,z.split(':')))
        for z_val in range(z_rng[0],z_rng[1]):
            ndviz_urls.append('https://viz-dev.boss.neurodata.io/#!{\'layers\':{\'' + ch + 
            '\':{\'type\':\'image\'_\'source\':\'boss://https://api.boss.neurodata.io/'
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

def process_params(request):
    q = request.GET
    
    #validation / data sanitization needed here because it's not being done in the form
    # x_rng_str = q.get('x')

    coll = q.get('coll')
    exp = q.get('exp')
    
    x = error_check_int_param(q.get('x'))
    y = error_check_int_param(q.get('y'))
    z = error_check_int_param(q.get('z'))
    channels = q.get('channels')
    channels = channels.split(',')
    
    return coll,exp,x,y,z,channels

def ndviz_url_list(request):
    coll,exp,x,y,z,channels = process_params(request)
    base_url, headers = get_boss_request(request,'')
    urls, channels_urls, channels_z = ret_ndviz_urls(base_url,coll,exp,x,y,z,channels)

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
        # tiff_stack_channel(request,coll,exp,x,y,z,ch)
    
    return render(request, 'synaptogram/tiff_url_list.html',{'urls': urls})

def create_voxel_rng(x,y,z):
    x_rng = list(map(int,x.split(':')))
    y_rng = list(map(int,y.split(':')))
    z_rng = list(map(int,z.split(':')))
    return x_rng,y_rng,z_rng

def tiff_stack_channel(request,coll,exp,x,y,z,channel):
    base_url, headers = get_boss_request(request,'')
    cu = ret_cut_urls(base_url,coll,exp,x,y,z,[channel])[0]

    x_rng,y_rng,z_rng = create_voxel_rng(x,y,z)

    headers_blosc = headers
    headers_blosc['Content-Type']='application/blosc-python'

    r_blosc=requests.get(cu,headers = headers_blosc)
    raw_data = blosc.decompress(r_blosc.content)
    data_mat = np.fromstring(raw_data, dtype='uint16')
    #if this is a time series, you need to reshape it differently
    z = np.reshape(data_mat,
                                (z_rng[1] - z_rng[0],
                                y_rng[1] - y_rng[0],
                                x_rng[1] - x_rng[0]),
                                order='C')
    
    response=HttpResponse(content_type='image/png')
    
    writer = png.Writer(width=z.shape[2], height=z.shape[1], bitdepth=16, greyscale=True)
    writer.write(response,z[0,:,:])
    
    #img = Image.fromarray(data_mat_reshape[0,:,:])
    #img.save(response, "PNG")
    return response

def sgram(request):
    coll,exp,x,y,z,channels = process_params(request)
    return plot_sgram(request,coll,exp,x,y,z,channels)

def plot_sgram(request,coll,exp,x,y,z,channels):
    base_url, headers = get_boss_request(request,'')
    cut_urls = ret_cut_urls(base_url,coll,exp,x,y,z,channels)
    
    x_rng,y_rng,z_rng = create_voxel_rng(x,y,z)

    num_ch = len(channels)
    num_z = z_rng[1] - z_rng[0]

    headers_blosc = headers
    headers_blosc['Content-Type']='application/blosc-python'

    fig=Figure(figsize=(10, 25), dpi= 150, facecolor='w', edgecolor='k')
    #fig=plt.figure(figsize=(10, 25), dpi= 150, facecolor='w', edgecolor='k')
    for ch_idx,cu in enumerate(cut_urls): #, exception_handler=exception_handler
        r_blosc=requests.get(cu,headers = headers_blosc)
        raw_data = blosc.decompress(r_blosc.content)
        data_mat = np.fromstring(raw_data, dtype='uint16')
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

def sgram_from_ndviz(request):
    url = request.GET.get('url')
    coll,exp,x,y,z = parse_ndviz_url(url)
    #get all the channels
    channels = get_all_channels(request,coll,exp)
    return plot_sgram(request,coll,exp,x,y,z,channels)