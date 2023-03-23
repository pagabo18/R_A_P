
import cv2
import mediapipe as mp
import open3d as o3d

from Patrones.singleton import *
from Hands.hand_detector import *
class ObjectViewer(metaclass=Singleton):
    ''' 
    Esta clase se encarga de visualizar los objetos 3D 
    '''

    def __init__(self, objectreadfile, makefullscreen=False, width=1366, height=768):
        '''
        iniciamos la instancia de ObjectViewer
        
        Parametros:
        objectreadfile: nombre del archivo 3D a visualizar
        makefullscreen: si es True, se ejecuta en pantalla completa
        width: El ancho de la ventana de visualización, por defecto es 1366.     
        height: La altura de la ventana de visualización, por defecto es 768.
        
        '''
        self.mesh = o3d.io.read_triangle_mesh(objectreadfile)
        self.mesh.compute_vertex_normals()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height)
        
        if makefullscreen:
            self.vis.set_full_screen(True)
        self.vis.add_geometry(self.mesh)
        self.vis.get_render_option().load_from_json("render_options.json")
        self.vis.poll_events()
        self.vis.update_renderer()
        self.zoomcounter = 0

    def vis_zoom(self, absZ):
        '''
        Zoom a la visualización a una profundidad absoluta dada.
        Parámetros:
        -----------
        absZ : 
            La profundidad absoluta a la que hacer zoom.
        
        '''
        self.vis.get_view_control().set_zoom(absZ)
        self.vis.poll_events()
        self.vis.update_renderer()

    def vis_rotate_reset(self):
        '''
        Reinicia la rotación de visualización
        
        '''
        self.vis.get_view_control().rotate(5, 0, xo=0.0, yo=0.0)

    def vis_rotate(self, deltaX, deltaY):
        '''
        Gire la visualización mediante un ángulo delta dado en los ejes x e y.

        Parámetros:
        -----------
        deltaX : 
            El ángulo de rotación en el eje x.
            
        deltaY : 
            El ángulo a girar en el eje y
        
        '''
        self.vis.get_view_control().rotate(-deltaX*10, -deltaY*10, xo=0.0, yo=0.0)
        self.vis.poll_events()
        self.vis.update_renderer()

    def vis_general_reset(self):
        """
        Restablezca la rotación de la visualización a sus valores predeterminados y aumente el contador de zoom en uno.
        """
        self.vis.get_view_control().rotate(5, 0, xo=0.0, yo=0.0)
        self.zoomcounter = self.zoomcounter + 1
        if self.zoomcounter > 1000:
            self.zoomcounter = 0
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        """
        corre continuamente la visualización hasta que se interrumpa.
        """
        
        while True:
            self.vis.get_view_control().rotate(5, 0, xo=0.0, yo=0.0)
            self.zoomcounter += 1
            if self.zoomcounter > 1000:
                self.zoomcounter = 0
            self.vis.poll_events()
            self.vis.update_renderer()
            
    def make_fullscreen(self,makefullscreen=False,hands_detection = None):
        """
        Maneja visualización a pantalla completa.

        Parámetros:
        -----------
        makefullscreen : 
            Si desea que la pantalla de visualización sea completa, el valor predeterminado es False.
        hands_detection : 
            Una instancia de HandDetector utilizada para mostrar los resultados de detección de manos en el modo de pantalla completa, por defecto es None.
        """
        if not makefullscreen:
                cv2.imshow('Hand Tracking', hands_detection.image)

            
    def close(self):
        '''
        Cierra la visualización de la ventana y destruye la instancia de ObjectViewer
        '''
        self.vis.destroy_window()
