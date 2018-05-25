import vtk

class VtkPointCloud:

    def __init__(self, zMin=-1, zMax=1, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        
        # actor
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        
        self.lines = vtk.vtkCellArray()

    def addPoint(self, point):          
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        
    def addLine(self, line_points):
        for point in line_points:
            pointId = self.vtkPoints.InsertNextPoint(point)
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()
        
        for i in range(len(line_points)-1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0,i) # the second 0 is the index of the Origin in the vtkPoints
            line.GetPointIds().SetId(1,i+1)
            
            self.lines.InsertNextCell(line)
            
    def processLines(self):
        # Add the lines to the dataset
        self.vtkPolyData.SetLines(self.lines)