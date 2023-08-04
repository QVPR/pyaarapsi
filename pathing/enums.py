from enum import Enum

class PS4_Buttons(Enum):
    X               = 0
    O               = 1
    Triangle        = 2
    Square          = 3
    LeftBumper      = 4
    RightBumper     = 5
    Share           = 6
    Options         = 7
    PS              = 8
    LeftStickIn     = 9
    RightStickIn    = 10
    LeftArrow       = 11
    RightArrow      = 12
    UpArrow         = 13
    DownArrow       = 14

class PS4_Triggers(Enum):
    LeftStickXAxis  = 0
    LeftStickYAxis  = 1
    LeftTrigger     = 2 # Released = 1, Pressed = -1
    RightStickXAxis = 3
    RightStickYAxis = 4
    RightTrigger    = 5 # Released = 1, Pressed = -1

class Safety_Mode(Enum):
    UNSET           = -1
    STOP            = 0
    SLOW            = 1
    FAST            = 2

class Command_Mode(Enum):
    STOP            = 0
    VPR             = 1
    SLAM            = 2
    ZONE_RETURN     = 3

class Lookahead_Mode(Enum):
    INDEX           = 0
    DISTANCE        = 1

class Return_Stage(Enum):
    UNSET           = 0
    DIST            = 1
    TURN            = 2
    DONE            = 3

class Reject_Mode(Enum):
    NONE            = 0
    STOP            = 1
    OLD             = 2
    OLD_50          = 150
    OLD_90          = 190