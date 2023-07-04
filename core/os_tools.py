#!/usr/bin/env python3

import os

def kill_screen(name):
    '''
    Kill screen

    Inputs:
    - name: str type, corresponding to full screen name as per 'screen -list'
    Returns:
    None
    '''
    os.system("if screen -list | grep -q '{sname}'; then screen -S '{sname}' -X quit; fi;".format(sname=name))

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
    os.system("screen -dmS '{sname}' bash -c '{scmd}; exec bash'".format(sname=name, scmd=cmd))