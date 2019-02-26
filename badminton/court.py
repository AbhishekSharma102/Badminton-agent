import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy

class RenderCourt():
    def __init__(self):
        self.vertices = (
            (260, 670, 0),
            (260, -670, 0),
            (-260, -670, 0),
            (-260, 670, 0) 
            )
        
        self.shuttle_vertices = (
            (10, -10, -10),
            (10, 10, -10),
            (-10, 10, -10),
            (-10, -10, -10),
            (10, -10, 10),
            (10, 10, 10),
            (-10, -10, 10),
            (-10, 10, 10)
            )
        
        self.shuttle_edges = (
            (0,1),
            (0,3),
            (0,4),
            (2,1),
            (2,3),
            (2,7),
            (6,3),
            (6,4),
            (6,7),
            (5,1),
            (5,4),
            (5,7)
            )
        
        self.shuttle_coord = [0, 0, 0]
        self.p1_coord = [0, 0]
        self.p2_coord = [0, 0]
    
    def IdentityMat44(self):
        return numpy.matrix(numpy.identity(4), copy=False, dtype='float32')
    
    def axes(self):
        glColor3f(1.0,0.0,0.0)
        glBegin(GL_LINES)
        
        glVertex3f(-40.0, 0.0, 0.0)
        glVertex3f(40.0, 0.0, 0.0)
    
        glVertex3f(40.0, 0.0, 0.0)
        glVertex3f(30.0, 10.0, 0.0)
    
        glVertex3f(40.0, 0.0, 0.0)
        glVertex3f(30.0, -10.0, 0.0)
        glEnd()
        glFlush()
    
        glColor3f(0.0,1.0,0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, -40.0, 0.0)
        glVertex3f(0.0, 40.0, 0.0)
    
        glVertex3f(0.0, 40.0, 0.0)
        glVertex3f(10.0, 30.0, 0.0)
     
        glVertex3f(0.0, 40.0, 0.0)
        glVertex3f(-10.0, 30.0, 0.0)
        glEnd()
        glFlush()
    
        glColor3f(0.0,0.0,1.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0 ,-40.0 )
        glVertex3f(0.0, 0.0 ,40.0 )
    
        glVertex3f(0.0, 0.0 ,40.0 )
        glVertex3f(0.0, 10.0 ,30.0 )
     
        glVertex3f(0.0, 0.0 ,40.0 )
        glVertex3f(0.0, -10.0 ,30.0 )
        glEnd()
        glColor3f(1.0,1.0,1.0)
    
    def boundary(self):
        glBegin(GL_LINES)
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[3])
        glVertex3fv(vertices[3])
        glVertex3fv(vertices[0])
        glEnd()
    
    def net(self):
        glBegin(GL_LINES)
        glVertex3fv((260, 0, 0))
        glVertex3fv((-260, 0, 0))
        glVertex3fv((260, 0, 0))
        glVertex3fv((260, 0, 152))
        glVertex3fv((260, 0, 152))
        glVertex3fv((-260, 0, 152))
        glVertex3fv((-260, 0, 152))
        glVertex3fv((-260, 0, 0))
        glEnd() 
    
    def shuttle(self):
        xcord, ycord, zcord = self.shuttle_cord[0], self.shuttle_cord[1], self.shuttle_cord[2]
        glColor3f(1.0,0.0,0.0)
        glBegin(GL_LINES)
        for edge in shuttle_edges:
            for vertex in edge:
                cord = list(shuttle_vertices[vertex])
                cord[0] += xcord
                cord[1] += ycord
                cord[2] += zcord
                glVertex3fv(cord)
        glEnd()
        glColor3f(1.0,1.0,1.0)
    
    def playerBase(self, xcord, ycord):
        glVertex3fv((xcord+10, ycord+10, 0))
        glVertex3fv((xcord+10, ycord-10, 0))
        glVertex3fv((xcord+10, ycord-10, 0))
        glVertex3fv((xcord-10, ycord-10, 0))
        glVertex3fv((xcord-10, ycord-10, 0))
        glVertex3fv((xcord-10, ycord+10, 0))
        glVertex3fv((xcord-10, ycord+10, 0))
        glVertex3fv((xcord+10, ycord+10, 0))
    
    def playerOne(self):
        glColor3f(0.0,0.0,1.0)
        glBegin(GL_LINES)
        self.playerBase(self.p1_cord[0], self.p1_cord[1])
        glEnd()
        glColor3f(1.0,1.0,1.0)
    
    def playerTwo(self):
        glColor3f(0.0,1.0,0.0)
        glBegin(GL_LINES)
        self.playerBase(self.p1_cord[0], self.p1_cord[1])
        glEnd()
        glColor3f(1.0,1.0,1.0)
    """
    def __init__(self):
        pygame.init()
        display = (800,600)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0]/display[1]), 0.1, 10000.0)
        
        view_mat = self.IdentityMat44()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -2000)
        glRotatef(-45, 1, 0, 0)
        glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)
        glLoadIdentity()
    
        while True:
            glPushMatrix()
            glLoadIdentity()
            glTranslatef(0, 0, 0)
            glRotatef(0, 0, 0, 0)
            glMultMatrixf(view_mat)
            glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self.boundary()
            self.net()
            self.shuttle()
            self.playerOne()
            self.playerTwo()
            glPopMatrix()
            pygame.display.flip()
            pygame.time.wait(10)
    """