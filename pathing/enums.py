from enum import Enum

class PS4_Pressed(Enum):
    X               = lambda msg: msg.buttons[  PS4_Buttons.X.value             ] > 0
    O               = lambda msg: msg.buttons[  PS4_Buttons.O.value             ] > 0
    Triangle        = lambda msg: msg.buttons[  PS4_Buttons.Triangle.value      ] > 0
    Square          = lambda msg: msg.buttons[  PS4_Buttons.Square.value        ] > 0
    LeftBumper      = lambda msg: msg.buttons[  PS4_Buttons.LeftBumper.value    ] > 0
    RightBumper     = lambda msg: msg.buttons[  PS4_Buttons.RightBumper.value   ] > 0
    LeftTrigger     = lambda msg: msg.buttons[  PS4_Buttons.LeftTrigger.value   ] > 0
    RightTrigger    = lambda msg: msg.buttons[  PS4_Buttons.RightTrigger.value  ] > 0
    Share           = lambda msg: msg.buttons[  PS4_Buttons.Share.value         ] > 0
    Options         = lambda msg: msg.buttons[  PS4_Buttons.Options.value       ] > 0
    PS              = lambda msg: msg.buttons[  PS4_Buttons.PS.value            ] > 0
    LeftStickIn     = lambda msg: msg.buttons[  PS4_Buttons.LeftStickIn.value   ] > 0
    RightStickIn    = lambda msg: msg.buttons[  PS4_Buttons.RightStickIn.value  ] > 0
    LeftArrow       = lambda msg: msg.axes[     PS4_Axes.ArrowsXAxis.value      ] > 0.5
    RightArrow      = lambda msg: msg.axes[     PS4_Axes.ArrowsXAxis.value      ] < -0.5
    UpArrow         = lambda msg: msg.axes[     PS4_Axes.ArrowsYAxis.value      ] > 0.5
    DownArrow       = lambda msg: msg.axes[     PS4_Axes.ArrowsYAxis.value      ] < -0.5

class PS4_Buttons(Enum):
    X               = 0
    O               = 1
    Triangle        = 2
    Square          = 3
    LeftBumper      = 4
    RightBumper     = 5
    LeftTrigger     = 6
    RightTrigger    = 7
    Share           = 8
    Options         = 9
    PS              = 10
    LeftStickIn     = 11
    RightStickIn    = 12

class PS4_Axes(Enum):
    LeftStickXAxis  = 0 # Left = 1, Right = -1
    LeftStickYAxis  = 1 # Up = 1, Down = -1
    LeftTrigger     = 2 # Released = 1, Pressed = -1
    RightStickXAxis = 3 # Left = 1, Right = -1
    RightStickYAxis = 4 # Up = 1, Down = -1
    RightTrigger    = 5 # Released = 1, Pressed = -1
    ArrowsXAxis     = 6 # Left = 1, Right = -1
    ArrowsYAxis     = 7 # Up = 1, Down = -1

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
    SPECIAL         = 4

class Experiment_Mode(Enum):
    UNSET           = 0
    INIT            = 1
    ALIGN           = 2
    DRIVE_PATH      = 3
    HALT1           = 4
    DRIVE_GOAL      = 5
    HALT2           = 6
    DONE            = 7

class Lookahead_Mode(Enum):
    INDEX           = 0
    DISTANCE        = 1

class Return_Stage(Enum):
    UNSET           = 0
    DIST            = 1
    TURN            = 2
    DONE            = 3

class Point_Shoot_Stage(Enum):
    UNSET           = 0
    INIT            = 1
    POINT           = 2
    SHOOT           = 3
    DONE            = 4

class Reject_Mode(Enum):
    NONE            = 0
    STOP            = 1
    OLD             = 2
    OLD_50          = 150
    OLD_90          = 190

class Technique(Enum):
    VPR             = 0
    SVM             = 1

class Save_Request(Enum):
    NONE            = 0
    CLEAR           = 1
    SET             = 2

class HDi(Enum):
    mInd            = 0     # match index
    tInd            = 1     # true index
    mDist           = 2     # match distance
    gt_class        = 3     # ground truth classification
    svm_class       = 4     # svm classification
    slam_x          = 5     # SLAM pose x
    slam_y          = 6     # SLAM pose y
    slam_w          = 7     # SLAM pose w
    robot_x         = 8     # robot wheel encoder pose x
    robot_y         = 9     # robot wheel encoder pose y
    robot_w         = 10    # robot wheel encoder pose w
    vpr_x           = 11    # VPR pose x
    vpr_y           = 12    # VPR pose y
    vpr_w           = 13    # VPR pose w
    dist            = 14    # distance since last point