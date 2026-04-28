# region: package imports
import os
import sys
import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem

# environment objects
from qvl.basic_shape import QLabsBasicShape
import pal.resources.rtmodels as rtmodels
import time
#endregion

def setup(
        initialPosition=[1.36, 1.311, 0.000],
        initialOrientation=[0, 0, -np.pi/2],
        rtModel=rtmodels.QCAR2
    ):

    # Try to connect to Qlabs
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    # Ensure that QLabs is running on your local machine
    print("Connecting to QLabs...")
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
        sys.exit()
        return
    print("Connected to QLabs")

    # Delete any previous QCar instances and stop any running spawn models
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(0.5)
    
    # region: QLabs title configuration
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('Color Space Experiment')
    # endregion
    
    # region: QCar spawn description

    # Spawn a QCar at the given initial pose
    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=[x for x in initialPosition],
        rotation=initialOrientation,
        waitForConfirmation=True
    )

    # Create a new camera view and attach it to the QCar
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([8.484, 1.973, 12.209], [-0, 0.748, 0.792])
    hqcar.possess()
    # endregion

    # region: Basic Shape for Depth Measurement
    cube0 = QLabsBasicShape(qlabs)
    cube0.spawn_id(actorNumber=0, 
                   location=[1.36,-15, 0], 
                   rotation=[0,0,0], 
                   scale=[1,1,2], 
                   configuration=cube0.SHAPE_CUBE, 
                   waitForConfirmation=True)
    cube0.set_material_properties(color=[1,0,0], 
                                  roughness=0.5, 
                                  metallic=False, 
                                  waitForConfirmation=True)
    
    # endregion

    # Start spawn model
    QLabsRealTime().start_real_time_model(rtModel)

    return hqcar


if __name__ == '__main__':
    # XXX Add processing of command line arguments
    setup()