#!/usr/bin/env python3
'''
Methods to download models from online repositories
'''
import os
import subprocess
from pyaarapsi.core.helper_tools import ask_yesnoexit
from pyaarapsi.vpr.classes import downloads

MODELS_DIR = downloads.__path__[0] + '/' #pylint: disable=W0212,E1101

def download_file(file_id, file_name):
    '''
    Shortcut to use script for downloading
    '''
    subprocess.run([MODELS_DIR + "downloader.sh", file_id, MODELS_DIR + file_name], check=True)

def download_netvlad_models(force=False):
    '''
    Download all relevant NetVLAD models
    '''
    dest_dir = os.path.join(MODELS_DIR)
    print(f"Path to netvlad model/s:\n\t<{dest_dir}>")
    if ask_yesnoexit("Auto-download pretrained NetVLAD model/s (takes around 2GB of space)? "
                     "(y)es/(n)o/(q)uit.", auto=5):
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA128.pth.tar"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA128.pth.tar")):
            print('Downloading mapillary_WPCA128.pth.tar')
            download_file("1v-MDvga__tblvZ9-m4Zfk6iem4c4Gjyh", "mapillary_WPCA128.pth.tar")
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA512.pth.tar"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA512.pth.tar")):
            print('Downloading mapillary_WPCA512.pth.tar')
            download_file("1D9V_k7VoTCbq6L0B8hVIqniV1vGFpCB8", "mapillary_WPCA512.pth.tar")
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar")):
            print('Downloading mapillary_WPCA4096.pth.tar')
            download_file("14fe_v2OYVJDa8qCiG1gcgf9I81qFRCGv", "mapillary_WPCA4096.pth.tar")
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar")):
            print('Downloading pittsburgh_WPCA128.pth.tar')
            download_file("1pnqUoREiTk-UJzU4N4n5doUm4dOIPxIU", "pittsburgh_WPCA128.pth.tar")
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar")):
            print('Downloading pittsburgh_WPCA512.pth.tar')
            download_file("1FGk77XsRo5ZRHHaxcCSmH_rhOdl0ByFs", "pittsburgh_WPCA512.pth.tar")
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar")):
            print('Downloading pittsburgh_WPCA4096.pth.tar')
            download_file("1TcF6Z2n7lxkf_9pXnEbJXmOzCMEljhjO", "pittsburgh_WPCA4096.pth.tar")
        print('Downloaded all pretrained NetVLAD models.')

def download_hybridnet_models(force=False):
    '''
    Download all relevant HybridNet models
    '''
    dest_dir = os.path.join(MODELS_DIR)
    print(f"Path to hybridnet model/s:\n\t<{dest_dir}>")
    if ask_yesnoexit("Auto-download pretrained HybridNet model/s (takes around 300MB of space)? "
                     "(y)es/(n)o/(q)uit.", auto=5):
        if force:
            try:
                os.remove(os.path.join(dest_dir, "HybridNet.caffemodel"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "HybridNet.caffemodel")):
            print('Downloading HybridNet.caffemodel')
            download_file("1_tzdDkM5TeyB37TEduyxWMi_T4CqQ4u4", "HybridNet.caffemodel")
        print('Downloaded all pretrained HybridNet models.')

def download_salad_models(force=False):
    '''
    Download all relevant SALAD models
    '''
    dest_dir = os.path.join(MODELS_DIR)
    print(f"Path to salad model/s:\n\t<{dest_dir}>")
    if ask_yesnoexit("Auto-download pretrained salad model/s (takes around 330MB of space)? "
                     "(y)es/(n)o/(q)uit.", auto=5):
        if force:
            try:
                os.remove(os.path.join(dest_dir, "dino_salad.ckpt"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "dino_salad.ckpt")):
            print('Downloading dino_salad.ckpt')
            download_file("1gOM7brNbkI_5i64Wc8SZf4N_X0OadCrT", "dino_salad.ckpt")
        print('Downloaded all pretrained SALAD models.')

def download_apgem_models(force=False):
    '''
    Download all relevant AP-GeM models
    '''
    dest_dir = os.path.join(MODELS_DIR)
    print(f"Path to apgem model/s:\n\t<{dest_dir}>")
    if ask_yesnoexit("Auto-download pretrained apgem model/s (takes around 590MB of space)? "
                     "(y)es/(n)o/(q)uit.", auto=5):
        if force:
            try:
                os.remove(os.path.join(dest_dir, "Resnet-101-AP-GeM.pt"))
            except OSError:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "Resnet-101-AP-GeM.pt")):
            print('Downloading Resnet-101-AP-GeM.pt')
            download_file("16V8ShtsbDHdBmbGWjfdWyVkNn36kkztB", "Resnet-101-AP-GeM.pt")
        print('Downloaded all pretrained AP-GeM models.')

def download_all_models(force=False):
    '''
    Download all relevant models
    '''
    download_netvlad_models(force)
    download_hybridnet_models(force)
    download_salad_models(force)
    download_apgem_models(force)

if __name__ == "__main__":
    download_all_models()
