#!/usr/bin/env python3
'''
OS tools, primarily for linux systems
'''
import os

def kill_screen(name):
    '''
    Kill screen

    Inputs:
    - name: str type, corresponding to full screen name as per 'screen -list'
    Returns:
    None
    '''
    os.system(f"if screen -list | grep -q '{name}'; then screen -S '{name}' -X quit; fi;")

def kill_screens(names):
    '''
    Kill all screens in list of screen names

    Inputs:
    - names: list of str type, elements corresponding to full screen name as per 'screen -list'
    Returns:
    None
    '''
    for name in names:
        kill_screen(name)

def exec_screen(name, cmd):
    '''
    Execute/start a screen session
    '''
    os.system(f"screen -dmS '{name}' bash -c '{cmd}; exec bash'")
