#!/usr/bin/env python3
'''
ANSI escape sequences
More available at: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
'''

C_I_RED     = '\033[91m'
C_I_GREEN   = '\033[92m'
C_I_YELLOW  = '\033[93m'
C_I_BLUE    = '\033[94m'
C_I_WHITE   = '\033[97m'
C_RESET     = '\033[0m'
C_CLEAR     = '\033[K'
C_UP_N      = '\033[%dA'
C_DOWN_N    = '\033[%dB'
C_RIGHT_N   = '\033[%dC'
C_LEFT_N    = '\033[%dD'
C_COLUMN    = '\033[%dG'
