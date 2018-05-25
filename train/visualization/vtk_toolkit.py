import vtk
import numpy as np

from visualization.vtkPointCloud import VtkPointCloud

def display_pc(pc):
        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(pc.vtkActor)
        renderer.SetBackground(.2, .3, .4)
        renderer.ResetCamera()

        # Setting a camera standing in the car's front view point
        # Setting the focal point as a point close to the in depth center of the point cloud
        camera = vtk.vtkCamera();  
        #camera.SetPosition(0, 0.2, -10)
        #camera.SetFocalPoint(12, -8, 69.333999633789063)

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderer.SetActiveCamera(camera)
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Begin Interaction
        renderWindow.Render()
        renderWindowInteractor.Start()

def create_pc(points, lines=None):
    pointCloud = VtkPointCloud()
    
    """
        Lines must be on the format:
            [
                [p1,p2, p...] - one line
                [p3,p4, p...] - another line
            ]
    """
    if lines is not None:
        for line in lines:
            pointCloud.addLine(line)
        
        pointCloud.processLines()

    for point in points:
        pointCloud.addPoint(point)

    return pointCloud

def velo2vtk_coord(coords):
    coords_a = np.array(coords)
    coords_a[:,[0, 1, 2]] = coords_a[:,[1, 2, 0]]
    
    return coords_a

def plot_pc(points=None, pc=None, lines=None):
    if points is None and pc is None:
        print('Nothing to plot :|')
    elif points is None:
        display_pc(pc)
    elif pc is None:
        pc_ = create_pc(points, lines)
        display_pc(pc_)
        