
import cv2
import mediapipe as mp
import open3d as o3d

from Patrones.singleton import *
from Hands.hand_detector import *

class ObjectViewer(metaclass=Singleton):

    def __init__(self, objectreadfile, makefullscreen=False, width=1366, height=768):
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

    def vis_rotate_view(self, angle_x, angle_y):
        self.vis.get_view_control().rotate(-angle_x*10, -angle_y*10, xo=0.0, yo=0.0)
        self.vis.poll_events()
        self.vis.update_renderer()

    def vis_zoom(self, absZ):
        self.vis.get_view_control().set_zoom(absZ)
        self.vis.poll_events()
        self.vis.update_renderer()

    def vis_rotate_reset(self):
        self.vis_rotate_view(5, 0)

    def vis_rotate(self, deltaX, deltaY):
        self.vis_rotate_view(deltaX, deltaY)

    def vis_general_reset(self):
        self.vis_rotate_view(5, 0)
        self.zoomcounter += 1
        if self.zoomcounter > 1000:
            self.zoomcounter = 0
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        while True:
            self.vis_rotate_view(5, 0)
            self.zoomcounter += 1
            if self.zoomcounter > 1000:
                self.zoomcounter = 0
            self.vis.poll_events()
            self.vis.update_renderer()
            
    def make_fullscreen(self,makefullscreen=False,hands_detection = None):
        if not makefullscreen:
                cv2.imshow('Hand Tracking', hands_detection.image)

            
    def close(self):
        self.vis.destroy_window()