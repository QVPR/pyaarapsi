#!/usr/bin/env python3

import os
import subprocess
import sys
import os
import imp # TODO: deprecated :-(
MODELS_DIR = imp.find_module("pyaarapsi")[1] + '/vpr_classes/downloads/'
from ..core.helper_tools import ask_yesnoexit

def download_file(id, file_name):
    subprocess.run([MODELS_DIR + "downloader.sh", id, MODELS_DIR + file_name])

def download_netvlad_models(force=False):
    global MODELS_DIR
    dest_dir = os.path.join(MODELS_DIR)
    print("Path to netvlad model/s:\n\t<%s>" % dest_dir)
    if ask_yesnoexit("Auto-download pretrained NetVLAD model/s (takes around 2GB of space)? (y)es/(n)o/(q)uit.", auto=5):
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA128.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA128.pth.tar")):
            print('Downloading mapillary_WPCA128.pth.tar')
            download_file("1v-MDvga__tblvZ9-m4Zfk6iem4c4Gjyh", "mapillary_WPCA128.pth.tar")

        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA512.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA512.pth.tar")):
            print('Downloading mapillary_WPCA512.pth.tar')
            download_file("1D9V_k7VoTCbq6L0B8hVIqniV1vGFpCB8", "mapillary_WPCA512.pth.tar")
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar")):
            print('Downloading mapillary_WPCA4096.pth.tar')
            download_file("14fe_v2OYVJDa8qCiG1gcgf9I81qFRCGv", "mapillary_WPCA4096.pth.tar")
       
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar")):
            print('Downloading pittsburgh_WPCA128.pth.tar')
            download_file("1pnqUoREiTk-UJzU4N4n5doUm4dOIPxIU", "pittsburgh_WPCA128.pth.tar")
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar")):
            print('Downloading pittsburgh_WPCA512.pth.tar')
            download_file("1FGk77XsRo5ZRHHaxcCSmH_rhOdl0ByFs", "pittsburgh_WPCA512.pth.tar")
       
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar")):
            print('Downloading pittsburgh_WPCA4096.pth.tar')
            download_file("1TcF6Z2n7lxkf_9pXnEbJXmOzCMEljhjO", "pittsburgh_WPCA4096.pth.tar")
        print('Downloaded all pretrained models.')

def download_hybridnet_models(force=False):
    global MODELS_DIR
    dest_dir = os.path.join(MODELS_DIR)
    print("Path to hybridnet model/s:\n\t<%s>" % dest_dir)
    if ask_yesnoexit("Auto-download pretrained HybridNet model/s (takes around 300MB of space)? (y)es/(n)o/(q)uit.", auto=5):
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "HybridNet.caffemodel"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "HybridNet.caffemodel")):
            print('Downloading HybridNet.caffemodel')
            download_file("1_tzdDkM5TeyB37TEduyxWMi_T4CqQ4u4", "HybridNet.caffemodel")

def download_salad_models(force=False):
    global MODELS_DIR
    dest_dir = os.path.join(MODELS_DIR)
    print("Path to salad model/s:\n\t<%s>" % dest_dir)
    if ask_yesnoexit("Auto-download pretrained salad model/s (takes around 330MB of space)? (y)es/(n)o/(q)uit.", auto=5):
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "dino_salad.ckpt"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "dino_salad.ckpt")):
            print('Downloading dino_salad.ckpt')
            download_file("1MH7vvvXmbjwMlU_bNx4CR4iFpu_fRYcY", "dino_salad.ckpt")

def download_all_models(force=False):
    
    download_netvlad_models(force)
    download_hybridnet_models(force)
    download_salad_models(force)

if __name__ == "__main__":
    download_all_models()